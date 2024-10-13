# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
from util import detector, calculate_auc_ap_fpr95
import cv2, time, os, wandb, gc, torch, math, argparse
from statistics import median
from tqdm.auto import tqdm
from dataset import *
from model import DLV3_CoroCL
from rpl_coroclcode.loss.PositiveEnergy import energy_loss, energy_loss_3d
from rpl_coroclcode.loss.CoroCL import ContrastLoss
import segmentation_models_pytorch_v2 as smp
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

class Trainer:
    def __init__(self, id_img_path, ood_img_path, val_img_path, val_mask_path, save_path, max_lr = 1e-3, epochs = 80, backbone = 'efficientnet-b6',
                 batch = 3, device = 'cuda', fac_noise_type = 'noise', real_noise_type = 'scraft', real_ood = False):
        self.id_img_path = id_img_path
        self.ood_img_path = ood_img_path
        self.val_img_path = val_img_path
        self.val_mask_path = val_mask_path
        self.save_path = save_path
        self.max_lr = max_lr
        self.backbone = backbone
        self.epochs = epochs
        self.batch = batch
        self.device = device
        self.fac_noise_type = fac_noise_type
        self.real_noise_type = real_noise_type
        self.ckpt = './ckpts/pretrain.pth'
        self.real_ood = real_ood
      
      
    def train_one_epoch(self):
        running_loss = 0
        ood_loss = 0
        self.model.train()
        prog = tqdm(enumerate(zip(self.train_loader_fac, self.train_loader_real)),total=len(self.train_loader_fac))
        for i, (data_fac, data_real)  in prog:
            #training phase
            image, mask ,mask_ood = data_fac
            image_fac = image.to(self.device)
            mask_fac = mask.to(self.device)
            mask_fac_ood = mask_ood.to(self.device)
            image ,mask_ood = data_real
            image_real = image.to(self.device)
            mask_real_ood = mask_ood.to(self.device)
            #forward
            prediction_fac, corocl_output_fac = self.model(image_fac)
            prediction_real, corocl_output_real = self.model(image_real)
            loss_dict = energy_loss_3d(logits=prediction_fac, targets=mask_fac.clone(), out_idx=100)
            inlier_loss = loss_dict["entropy_part"]
            contras_loss = self.corocl_loss(corocl_output_fac, mask_fac_ood, prediction_fac, corocl_output_real, mask_real_ood, prediction_real)
            outlier_loss = loss_dict["energy_part"] * 0.25
            if self.real_ood:
                outlier_msk = (mask_real_ood.clone()==1)
                if torch.sum(outlier_msk) > 0:
                    logits = prediction_real.flatten(start_dim=2).permute(0, 2, 1)
                    real_loss = torch.nn.functional.relu(torch.log(torch.sum(torch.exp(logits),
                                    dim=2))[outlier_msk.flatten(start_dim=1)]).mean()*0.25
                    outlier_loss += real_loss            
            in_n_con_loss = inlier_loss+contras_loss
            loss = outlier_loss+in_n_con_loss
            #backward
            loss.backward()
            self.optimizer.step() #update weight          
            self.optimizer.zero_grad() #reset gradient
            self.scheduler.step() 
            running_loss += loss.item()
            ood_loss += outlier_loss.item()
        #self.weight = math.tanh(ood_loss/len(self.train_loader_fac))
        return running_loss/len(self.train_loader_fac), ood_loss/len(self.train_loader_fac)
    
    def val_one_epoch(self):
        self.model.eval()
        auc_list = []
        ap_list = []
        fpr95_list = []
        for idx in tqdm(range(len(os.listdir(self.val_img_path)))):
            img_path = os.path.join(self.val_img_path, sorted(os.listdir(self.val_img_path))[idx])
            mask_path = os.path.join(self.val_mask_path, sorted(os.listdir(self.val_mask_path))[idx])
            y_true, y_scores = detector(self.model, img_path, mask_path, rba = False)
            # Calculate AUC, AP, and FPR95
            auc, ap, fpr95 = calculate_auc_ap_fpr95(y_true, y_scores)
            auc_list.append(auc)
            ap_list.append(ap)
            fpr95_list.append(fpr95)
        return sum(auc_list) / len(auc_list), sum(ap_list) / len(ap_list), sum(fpr95_list) / len(fpr95_list)
    
    def fit(self):
        self.train_loader_fac, self.train_loader_real = self.get_dl()
        self.model = DLV3_CoroCL(backbone=self.backbone, ckpt=self.ckpt, classes=self.classes).to(self.device)
        self.corocl_loss = ContrastLoss(1)
        non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(non_frozen_parameters, lr=self.max_lr, weight_decay=1e-4)
        total_step = self.epochs*len(self.train_loader_fac)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.max_lr, total_steps=total_step, pct_start=1000/total_step)
        self.model.to(self.device)
        for e in range(self.epochs):
            train_loss, train_ood_loss = self.train_one_epoch()
            auc, ap, fpr95 = self.val_one_epoch()
            wandb.log({"train_loss": train_loss,
                    "train_ood_loss": train_ood_loss,
                    
                    "AUC": auc,
                    "AP": ap,
                    "FPR": fpr95,
                    })
            self.save_model(e)
    def get_dl(self):
        height = 512
        width = 512
        t_train = A.Compose([A.Resize(height, width, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                     A.RandomBrightnessContrast((-0.5,0.5),(-0.5,0.5)), A.GaussNoise()])
        X_fac_train = glob.glob(os.path.join(self.id_img_path, '*jpg'))
        X_real_train = glob.glob(os.path.join(self.ood_img_path, '*jpg'))
        train_set_fac = FacadesDataset(X_fac_train, normal=False, noise_type=self.fac_noise_type)
        train_loader_fac = DataLoader(train_set_fac, batch_size=self.batch, shuffle=True, drop_last=True)

        train_set_real = buildingDataset(X_real_train, noise_type=self.real_noise_type)
        train_loader_real = DataLoader(train_set_real, batch_size=self.batch, shuffle=True, drop_last=True)

        return train_loader_fac, train_loader_real
        
    def save_model(self, e):
        if os.path.isdir(self.save_path)==False:
            os.mkdir(self.save_path)
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{e}.pth"))
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_img_path', type=str, default='./datasets/facades/origin', help='source image path')
    parser.add_argument('--ood_img_path', type=str, default='./datasets/HRSV/train/pure_tile', help='target image path')
    parser.add_argument('--val_img_path', type=str, default='./datasets/val_test/val/image/', help='validation image path')
    parser.add_argument('--val_mask_path', type=str, default='./datasets/val_test/val/GT/', help='validation label path')
    parser.add_argument('--save_path', type=str, default='./ckpts', help='checkpoint save path')
    parser.add_argument('--max_lr', type=float, default=1e-4, help='max learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='max epoch')
    parser.add_argument('--batch', type=int, default=3, help='batch size')
    parser.add_argument('--fac_noise_type', type=str, default='noise', help='inlier outlier exposure module')
    parser.add_argument('--real_noise_type', type=str, default='scraft', help='outlier outlier exposure module')
    
    args = parser.parse_args()
    model = Trainer(id_img_path = args.id_img_path,
                    ood_img_path = args.ood_img_path,
                    val_img_path = args.val_img_path, 
                    val_mask_path = args.val_mask_path,
                    save_path = args.save_path,
                    max_lr = args.max_lr,
                    epochs = args.epochs,
                    batch = args.batch,
                    fac_noise_type = args.fac_noise_type,
                    real_noise_type = args.real_noise_type
                    )
    with wandb.init(project="RPL"):
        model.fit()

    
if __name__ == '__main__':
    main()

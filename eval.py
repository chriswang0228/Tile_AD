import os, torch
from tqdm import tqdm
from dataset import *
from model import DLV3_CoroCL
from util import detector, calculate_auc_ap_fpr95

device = 'cuda'
model = DLV3_CoroCL(ckpt = None).to(device)
model.load_state_dict(torch.load('./ckpts/ckpt.pth'))
model.eval()
val_img_path = './datasets/val_test/test/image/'
val_mask_path = './datasets/val_test/test/GT/'
auc_list = []
ap_list = []
fpr95_list = []
for idx in tqdm(range(len(os.listdir(val_img_path)))):
    img_path = os.path.join(val_img_path, sorted(os.listdir(val_img_path))[idx])
    mask_path = os.path.join(val_mask_path, sorted(os.listdir(val_mask_path))[idx])
    y_true, y_scores = detector(model, img_path, mask_path, rba=False)
    # Calculate AUC, AP, and FPR95
    auc, ap, fpr95 = calculate_auc_ap_fpr95(y_true, y_scores)
    auc_list.append(auc)
    ap_list.append(ap)
    fpr95_list.append(fpr95)

print("AUC:", sum(auc_list) / len(auc_list))
print("Average Precision:", sum(ap_list) / len(ap_list))
print("FPR at TPR 95%:", sum(fpr95_list) / len(fpr95_list))

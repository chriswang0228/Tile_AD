import cv2, torch, kornia
import numpy as np
from PIL import Image
from dataset import *
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch.nn as nn

def calculate_auc_ap_fpr95(y_true, y_scores):
    """
    Calculate AUC, AP, and FPR95 for image anomaly detection.

    Args:
    - y_true (numpy array): True labels (1 for anomalies, 0 for normal).
    - y_scores (numpy array): Anomaly scores or model predictions.

    Returns:
    - auc (float): Area Under the ROC Curve.
    - ap (float): Average Precision.
    - fpr95 (float): False Positive Rate at a True Positive Rate of 95%.
    """
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # Calculate FPR at TPR of 95%
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.searchsorted(tpr, 0.95, side="right")
    fpr95 = fpr[idx] if idx < len(fpr) else fpr[-1]

    return auc, ap, fpr95

def detector(model, img_path, mask_path, segformer = False, rba = False, sizes=[384, 384]):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, sizes)
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, sizes, cv2.INTER_NEAREST)
    mask[mask>0]=1
    img = Image.fromarray(img)
    t = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_tnesor = t(img).to('cuda').unsqueeze(0)
    if segformer:
        out = model(img_tnesor)
        prediction_fac = nn.functional.interpolate(out,
                    size=sizes, # (height, width)
                    mode='bilinear',
                    align_corners=False)
    else:
        prediction_fac, corocl_output_fac = model(img_tnesor)
    if rba:
        energy = -prediction_fac.tanh().sum(dim=1) 
    else:
        energy = -torch.logsumexp(prediction_fac, dim=1)
        
    energy = energy[0].cpu().detach()
    energy = energy.numpy()
    y_true = mask.reshape(-1)  # 1 for anomalies, 0 for normal
    y_scores = energy.reshape(-1)  # Anomaly scores or model predictions
    
    return y_true, y_scores

def supervised_detector(model, img_path, mask_path):
    m = nn.Softmax(dim=1)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (512, 512), cv2.INTER_NEAREST)
    mask[mask>0]=1
    img = Image.fromarray(img)
    t = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_tnesor = t(img).to('cuda').unsqueeze(0)
    #output = m(pred)[0][1]
    output = model(img_tnesor).squeeze().softmax(dim=0)[1]
    #output = torch.argmax(output, dim=0)
    y_true = mask.reshape(-1)  # 1 for anomalies, 0 for normal
    y_scores = output.reshape(-1).cpu().detach().numpy()  # Anomaly scores or model predictions
    
    return y_true, y_scores

def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)
def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)
def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target
def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target
def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target

def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def get_mean_std(batch_size, dev):
    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]
    mean = [
        torch.as_tensor(mean, device=dev)
        for i in range(4)
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(std, device=dev)
        for i in range(4)
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std

def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target
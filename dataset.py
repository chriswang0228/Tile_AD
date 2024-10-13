import os, cv2, torch, math, random
import numpy as np
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Tuple
from torchvision.transforms import v2
from perlin_noise import PerlinNoise

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))

def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points

def batch_gen(img, mask, noise_type = 'scraft', real=True):
    transforms = v2.Compose([
    v2.ToTensor(),
    v2.RandomGrayscale(1)
    ])
    h,w = np.array(mask).shape[:2]
    spalling_mask = np.zeros((h, w), np.uint8)
    spalling_color = int(random.randint(50, 150))

    noise = PerlinNoise(octaves=333)
    xpix, ypix = 100, 100
    p_noise = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]
    p_noise = (np.array(p_noise)+1)*128
    p_noise = np.stack([cv2.resize(p_noise, (w, h)), cv2.resize(p_noise, (w, h)), cv2.resize(p_noise, (w, h))], axis = 2).astype(np.uint8)

    gray_img = (transforms(img).permute([1,2,0]).numpy()*255).astype(np.uint8)
    alpha = 0.5
    b = random.uniform(0.5, 1)
    gray_img = gray_img*alpha + p_noise*(1-alpha)
    gray_img = (gray_img*b).astype(np.uint8)

    for _ in range(10):
        y = int(np.clip(np.random.normal(h/2.,h/5.),0,h-1))
        x = int(np.clip(np.random.normal(w/2.,w/5.),0,w-1))
        point = (x,y)
        vertices = generate_polygon(center=point,
                                avg_radius=random.randint(20, 40),
                                irregularity=1,
                                spikiness=0.3,
                                num_vertices=8)
        points = np.array(vertices, np.int32)
        cv2.fillPoly(spalling_mask, pts=[points], color=1)
    if real:
        spalling_mask[np.array(mask)!=1]=0
    else:
        spalling_mask[np.array(mask)!=2]=0
    np_img, noisy_img, gray = np.array(img), np.array(img), np.array(img)
    np_img[spalling_mask==1]=(spalling_color, spalling_color, spalling_color)

    noisy_img[spalling_mask==1]=(0, 0, 0)
    p_noise[spalling_mask!=1]=(0, 0, 0)
    noisy_img+=p_noise

    gray[spalling_mask==1]=(0, 0, 0)
    gray_img[spalling_mask!=1]=(0, 0, 0)
    gray+=gray_img
    if noise_type == 'scraft':
        return gray, spalling_mask
    elif noise_type == 'noise':
        return noisy_img, spalling_mask
    else:
        return np_img, spalling_mask
    
class buildingDataset(Dataset):
    
    def __init__(self, X, transform = None, noise_type = 'scraft', return_norm = False):
        self.X = X
        self.transform = transform
        self.noise_type = noise_type
        self.return_norm = return_norm
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = Image.open(self.X[idx])
        mask = Image.open(self.X[idx][:-4]+'.png')
        img, mask = batch_gen(image, mask, noise_type = self.noise_type, real = True)
        img = cv2.resize(img, (384, 384))
        image = image.resize([384, 384])
        mask = cv2.resize(mask, (384, 384), interpolation=cv2.INTER_NEAREST)#*100
        if self.transform is not None:
                aug = self.transform(image=img, mask=mask)
                img = Image.fromarray(aug['image'])
                mask = aug['mask']

        if self.transform is None:
                img = Image.fromarray(img)
        t = T.Compose([T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = t(img)
        image = t(image)
        if self.return_norm:
            return img, image, torch.from_numpy(mask).long()
        else:
            return img, torch.from_numpy(mask).long()
                
class valDataset(Dataset):
    
    def __init__(self, img_path, mask_path, transform):
        super().__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir(self.img_path))
    
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_path, os.listdir(self.img_path)[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_path, os.listdir(self.img_path)[idx][:-4]+'_label.png'), 0)
        mask[mask==255]=1

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']


        if self.transform is None:
            img = Image.fromarray(img)
        
        t = T.Compose([T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = t(img)
        mask = torch.from_numpy(mask).long()
        return img, mask
    
class FacadesDataset(Dataset):
    
    def __init__(self, X, transform=None, normal=True, noise_type='scraft', return_norm = False, custom_class = False):
        self.X = X
        self.transform = transform
        self.normal = normal
        self.noise_type = noise_type
        self.return_norm = return_norm
        self.custom_class = custom_class
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        if self.normal:
            img = cv2.imread(self.X[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = np.array(Image.open(self.X[idx][:-4]+'.png'))

            if self.transform is not None:
                aug = self.transform(image=img, mask=mask)
                img = Image.fromarray(aug['image'])
                mask = aug['mask']


            elif self.transform is None:
                img = Image.fromarray(img)
            
            t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
            img = t(img)
            mask = torch.from_numpy(mask).long()
                
            return img, mask
        else:
            image = Image.open(self.X[idx])
            if self.custom_class:
                mask = Image.open(self.X[idx][:-4]+'_labelCustom.png')
            else:
                mask = Image.open(self.X[idx][:-4]+'.png')
            img, mask_ood = batch_gen(image, mask, noise_type = self.noise_type, real = False)
            img = cv2.resize(img, (384, 384))
            image = image.resize([384, 384])
            mask_ood = cv2.resize(mask_ood, (384, 384), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(np.array(mask), (384, 384), interpolation=cv2.INTER_NEAREST)
            mask[mask_ood==1] = 100
            if self.transform is not None:
                aug = self.transform(image=img, mask=mask, mask_ood=mask_ood)
                img = Image.fromarray(aug['image'])
                mask = aug['mask']
                mask_ood = aug['mask_ood']


            elif self.transform is None:
                img = Image.fromarray(img)
            t = T.Compose([T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            img = t(img)
            image = t(image)
            if self.return_norm:
                return img, image, torch.from_numpy(mask).long(), torch.from_numpy(mask_ood).long()
            else:
                return img, torch.from_numpy(mask).long(), torch.from_numpy(mask_ood).long()
    
        
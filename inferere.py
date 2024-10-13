import os, torch, argparse
from dataset import *
from model import DLV3_CoroCL

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='./datasets/val_test/test/image/', help='image path')
parser.add_argument('--save_path', type=str, default='./output', help='predictions save path')
parser.add_argument('--ckpt', type=str, default='./ckpts/ckpt.pth', help='checkpoint path')

args = parser.parse_args()

device = 'cuda'
model = DLV3_CoroCL(ckpt = None).to(device)
model.load_state_dict(torch.load(args.ckpt))
model.eval()
val_img_path = args.img_path

for idx in range(len(os.listdir(val_img_path))):
    img = cv2.imread(os.path.join(val_img_path, sorted(os.listdir(val_img_path))[idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = Image.fromarray(img)
    t = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_tnesor = t(img).to(device).unsqueeze(0)
    prediction_fac, corocl_output_fac = model(img_tnesor)
    energy = -torch.logsumexp(prediction_fac, dim=1)
    energy = energy[0].cpu().detach()
    energy -= energy.min()
    energy /= energy.max()
    energy = energy.numpy()*255.0
    cv2.imwrite(os.path.join(args.save_path, f'{sorted(os.listdir(val_img_path))[idx]}'), energy)
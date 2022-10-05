'''
The code was modified based on the repository https://github.com/zhilin007/FFA-Net [1].
[1] X. Qin, Z. Wang, Y. Bai, X. Xie, H. Jia, Ffa-net: Feature fusion attention network for single image dehazing, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 34, 2020, pp. 11908–11915.
'''

import os,argparse
import numpy as np
from PIL import Image
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--test_imgs',type=str,default='../test_comparison_for_cvpr/test_in',help='Test imgs folder')
parser.add_argument('--net', type=str, default='default_pa')
opt=parser.parse_args()
gps=6
blocks=19
img_dir=opt.test_imgs+'/'
output_dir = os.path.join(os.getcwd(), 'pred_output')
print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_dir = 'trained_models/our_model.pk'
device='cuda' if torch.cuda.is_available() else 'cpu'
ckp=torch.load(model_dir,map_location=device)
models_={
    'default_pa':MODEL_PA(gps=gps,blocks=blocks),
}
net = models_[opt.net]
net=nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()
for im in os.listdir(img_dir):
    if('lan' in im):
        continue 
    print(f'\r {im}',end='',flush=True)
    input_img = Image.open(img_dir+im)
    
    print(np.array(input_img).shape)
    if('.png' in im):
        input_img = input_img.convert('RGB')
    input_img_= tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(input_img)[None,::]
    with torch.no_grad():
        pred = net(input_img_)
    ts=torch.squeeze(pred.clamp(0,1).cpu())
    vutils.save_image(ts,os.path.join(output_dir, im.replace('.png', '_res.png')))

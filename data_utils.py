'''
The code was modified based on the repository https://github.com/zhilin007/FFA-Net [1] to support 3 datasets: RESIDE dataset, Rui's dataset [2], our collected MistAndRaindrop dataset
[1] X. Qin, Z. Wang, Y. Bai, X. Xie, H. Jia, Ffa-net: Feature fusion attention network for single image dehazing, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 34, 2020, pp. 11908–11915.
[2] R. Qian, R.T. Tan, W. Yang, J. Su, J. Liu, Attentive generative adversarial network for raindrop removal from a single image, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 2482–2491.
'''

import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
from glob import glob
# import cv2
BS=opt.bs
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size

# def tensorShow(tensors,titles=None):
#         '''
#         t:BCWH
#         '''
#         fig=plt.figure()
#         for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
#             img = make_grid(tensor)
#             npimg = img.numpy()
#             ax = fig.add_subplot(211+i)
#             ax.imshow(np.transpose(npimg, (1, 2, 0)))
#             ax.set_title(tit)
#         plt.show()

class Dataset_Cls(data.Dataset):
    def __init__(self,path,train, size=crop_size,format='.png', reading_type = 'generator'):
        super(Dataset_Cls,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs = []
        self.clear_imgs = []
        self.use_RESIDE = True if (len(path.split('RESIDE')) > 1) else False
        print('Use RESIDE? : {}'.format(self.use_RESIDE))
        if(self.use_RESIDE):                        # RESIDE dataset
            self.haze_imgs_dir=os.listdir(os.path.join(path,'haze'))
            self.haze_imgs=[os.path.join(path,'haze',img) for img in self.haze_imgs_dir]
            self.clear_dir=os.path.join(path,'clear')
        elif('rui' in path):                        # Rui's dataset
            self.haze_imgs_dir=os.listdir(os.path.join(path,'data'))
            self.haze_imgs=[os.path.join(path,'data',img) for img in self.haze_imgs_dir]
            self.clear_dir=os.path.join(path,'gt')
        else:                                       # Our MistAndRaindrop dataset
            gt_paths = glob(path + '/*lan*')
            print('DEBUG: {}'.format(len(gt_paths)))
            for gt_path in gt_paths:
                scene_index = (gt_path.split('/')[-1]).split('.')[0]
                corres_in_paths = glob(path + '/' + scene_index + '.*')
                for corres_in_path in corres_in_paths:
                    if((corres_in_path.split('/')[-1]).split('-')[-1][:3] != 'lan'):
                        self.clear_imgs.append(gt_path)
                        self.haze_imgs.append(corres_in_path)
            assert len(self.clear_imgs) == len(self.haze_imgs), 'The number of clear images mismatch with that of degraded images.'

        self.reading_type = reading_type
        self.haze_data = []
        self.clear_data = []
        if(self.reading_type == 'whole'):
            for i, haze_img in enumerate(self.haze_imgs):
                haze = Image.open(haze_img)
                if(self.use_RESIDE or 'rui' in haze_img):
                    id=haze_img.split('/')[-1].split('_')[0]
                    clear_name=id + '_clean' + self.format if (not self.use_RESIDE) else id + self.format
                    clear=Image.open(os.path.join(self.clear_dir,clear_name))
                    haze = haze.crop((0,0,720,480)) if(not self.use_RESIDE) else haze
                    clear = clear.crop((0,0,720,480)) if(not self.use_RESIDE) else clear
                    clear=tfs.CenterCrop(haze.size[::-1])(clear) if(self.use_RESIDE) else clear
                else:
                    clear_path = self.clear_imgs[i]
                    clear=Image.open(clear_path)
                self.haze_data.append(haze)
                self.clear_data.append(clear)

    def __getitem__(self, index):
        if(self.reading_type == 'generator'):
            haze=Image.open(self.haze_imgs[index])
            if isinstance(self.size,int):
                while haze.size[0]<self.size or haze.size[1]<self.size :
                    index=random.randint(0,20000)
                    haze=Image.open(self.haze_imgs[index])
            img=self.haze_imgs[index]
            if(self.use_RESIDE or 'rui' in img):        # RESIDE dataset or Rui's dataset
                id=img.split('/')[-1].split('_')[0]
                clear_name=id + '_clean' + self.format if (not self.use_RESIDE) else id + self.format
                clear=Image.open(os.path.join(self.clear_dir,clear_name))
            else:                                       # Our MistAndRaindrop dataset
                clear_path = self.clear_imgs[index]
                clear=Image.open(clear_path)
            if(self.use_RESIDE or 'rui' in img):
                haze = haze.crop((0,0,720,480)) if(not self.use_RESIDE) else haze
                clear = clear.crop((0,0,720,480)) if(not self.use_RESIDE) else clear
            clear=tfs.CenterCrop(haze.size[::-1])(clear) if(self.use_RESIDE) else clear
        elif(self.reading_type == 'whole'):
            haze = self.haze_data[index]
            clear = self.clear_data[index]
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            # print(haze.size)
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
            if(self.use_cam):
                cam_data = cam_data[i:i+h, j:j+w]
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB"))
        # haze=tfs.ToTensor()(haze)
        # clear=tfs.ToTensor()(clear)
        return haze,clear
    def augData(self,data,target, cam_data = None):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            rand_ver=random.randint(0,1)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            data=tfs.RandomVerticalFlip(rand_ver)(data)
            target=tfs.RandomVerticalFlip(rand_ver)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        # print(data.size())
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)

# from prefetch_generator import BackgroundGenerator
# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

import os
pwd=os.getcwd()
print(pwd)
ITS_train_loader = None
ITS_test_loader = None
OTS_train_loader = None
OTS_test_loader = None
rui_train_loader = None
rui_test_loader = None
if (opt.trainset[:3] == 'its' or opt.trainset[:3] == 'ots'):
    path='./datasets'

    ITS_train_loader=DataLoader(dataset=Dataset_Cls(path+'/RESIDE/ITS',train=True,size=crop_size),batch_size=BS,shuffle=True)
    ITS_test_loader=DataLoader(dataset=Dataset_Cls(path+'/RESIDE/SOTS/indoor',train=False,size='whole img'),batch_size=1,shuffle=False)

    OTS_train_loader=DataLoader(dataset=Dataset_Cls(path+'/RESIDE/OTS',train=True,format='.jpg'),batch_size=BS,shuffle=True)
    OTS_test_loader=DataLoader(dataset=Dataset_Cls(path+'/RESIDE/SOTS/outdoor',train=False,size='whole img',format='.png'),batch_size=1,shuffle=False)

else:
    path = './datasets' # For Rui's dataset or our MistAndRaindrop dataset

    our_train_loader=DataLoader(dataset=Dataset_Cls(path+'/MistAndRaindrop/train',train=True,format='.png', reading_type='generator'),batch_size=BS,shuffle=True)
    our_test_loader=DataLoader(dataset=Dataset_Cls(path+'/MistAndRaindrop/val',train=False,size='whole img',format='.png'),batch_size=1,shuffle=False)

    # rui_train_loader=DataLoader(dataset=Dataset_Cls(path+'/rui_dataset/train', train=True,format='.png'),batch_size=BS,shuffle=True, num_workers=0)
    # rui_test_loader=DataLoader(dataset=Dataset_Cls(path+'/rui_dataset/test_a', train=False,size='whole img',format='.png'),batch_size=1,shuffle=False)

if __name__ == "__main__":
    pass

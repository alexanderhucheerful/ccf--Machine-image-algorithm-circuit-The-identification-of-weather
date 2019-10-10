from torch.utils.data import Dataset
from torchvision import transforms as T 
from config import config
from PIL import Image 
from dataset.aug import *
from itertools import chain 
from glob import glob
from tqdm import tqdm
import random 
import numpy as np 
import pandas as pd 
import os 
import cv2
import torch 

#1.set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

#2.define dataset
class ChaojieDataset(Dataset):
    def __init__(self,label_list,transforms=None,train=True,test=False):
        self.test = test 
        self.train = train 
        imgs = []
        if self.test:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"]))
            self.imgs = imgs 
        else:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"],row["label"]))
            self.imgs = imgs
        if transforms is None:
            if self.test or not self.train:
                self.transforms = T.Compose([
                    T.Resize((config.img_weight,config.img_height)),
                    T.ToTensor(),
                    T.Normalize(mean = [0.485,0.456,0.406],
                                std = [0.229,0.224,0.225])])
            else:
                self.transforms  = T.Compose([
                    T.Resize((config.img_weight,config.img_height)),
                    T.RandomRotation(30),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomAffine(45),
                    T.ToTensor(),
                    T.Normalize(mean = [0.485,0.456,0.406],
                                std = [0.229,0.224,0.225])])
        else:
            self.transforms = transforms
    def __getitem__(self,index):
        if self.test:
            filename = self.imgs[index]
            #img = cv2.imread(filename)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            img = Image.open(filename)
            x = np.array(img).shape[2]
            print(x)
            if x == 4:
                img = Image.open(filename).convert("RGB")
            img = self.transforms(img)
            return img,filename
        else:
            filename,label = self.imgs[index] 
            #img = cv2.imread(filename)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #img = Image.open(filename)
            try:
                img = Image.open(filename).convert("RGB")
            except:
                filename = self.imgs[index-1]
                img = Image.open(filename).convert("RGB")
                
            print(np.array(img).shape)
            img = self.transforms(img)
            return img,label
    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label

def get_files(root,root1,mode):
    #for test
    if mode == "test":
        files = []
        #for img in os.listdir(root):
        #x = pd.read_csv(root)  
        #all_data_paths,all_data_path = [],[]
        #all_data_paths = list(x.iloc[:,0])
        #for i in all_data_paths:
            #all_data_path.append(root1+all_data_paths[i]) 
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files
    elif mode != "test": 
        #for train and val     
        x = pd.read_csv(root)  
        all_data_paths,labels,all_data_path = [],[],[]
        #image_folders = list(map(lambda x:root+x,os.listdir(root)))
        #all_images = list(chain.from_iterable(list(map(lambda x:glob(x+"/*"),image_folders))))
        print("loading train dataset")
        #for file in tqdm(all_images):

        all_data_paths = list(x.iloc[:,0])
        for i in range(len(all_data_paths)):
            #print(all_data_paths[i])
            temp = str(root1+str(all_data_paths[i]))
            #print(temp)
            all_data_path.append(temp) 
            #print(all_data_path[i])
        labels = list(x.iloc[:,1])
        all_files = pd.DataFrame({"filename":all_data_path,"label":labels})
        return all_files
    else:
        print("check the mode please!")
    

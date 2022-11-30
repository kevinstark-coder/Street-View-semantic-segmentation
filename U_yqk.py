'''
Project name: Unet
authur: Qiankun Yang
date 2022/7/29
---------------------
Following questions:
optimizer, adam vs SGD
'''
#Adam SGD
import torch.optim as optim
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import h5py
import cv2.cv2 as cv
import torch.nn as nn
import torchvision.transforms as T

def seed_torch(seed=1):
    """
    set seed of random, making the program repeatable.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                                    #为CPU设置种子用于生成随机数，以使得结果是确定的
    # if you use GPU, the following lines are needed.
    torch.cuda.manual_seed(seed)                               #为当前GPU设置随机种子；
    torch.cuda.manual_seed_all(seed) # if using multi-GPU.     #如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    torch.backends.cudnn.benchmark = False                      #初始寻找最优卷积
    torch.backends.cudnn.deterministic = True                  #继承上次卷积算法

def decode_segmap(images, color_code):
    for i, image in enumerate(images):
        r = np.zeros_like(image)
        g = np.zeros_like(image)
        b = np.zeros_like(image)
        for lab in range(34):
            r[image == lab] = color_code[lab, 0]
            g[image == lab] = color_code[lab, 1]
            b[image == lab] = color_code[lab, 2]
        images[i] = np.stack([r,g,b],axis=2)
    return images

transform = T.Compose([
  #  T.CenterCrop(128),
    T.ToTensor(),
    #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#
# class Cityscape(Dataset):
#     def __init__( self, root_dir, transform = None):
#         super(Cityscape,self).__init__()
#         self.root_dir= root_dir
#         self.data= self.get_data(root_dir)
#         self.transform= transform
#     def get_data(self,path):
#         with h5py.File(path, 'r') as file_object:
#             imgs = np.array(file_object["rgb"])
#             labs = np.array(file_object["seg"])
#             print(type(imgs))
#             print(imgs.shape)
#             # for img in imgs:
#             #     img = cv.resize(imgs,(256,256))
#             images = np.zeros((2975, 256, 256, 3))
#             for i,img in enumerate(imgs):
#                 images[i] = cv.resize(img, (256, 256))
#             images.astype(np.uint8)
#             labels = np.zeros((2975, 256, 256))
#             for i,lab in enumerate(labs):
#                 labels[i] = cv.resize(lab,(256,256))
#             labels.astype(np.uint8)
#             return images,labels
#     def __getitem__(self, index):
#             imgs,labs= self.data
#             imgs=imgs[index]
#             labs=labs[index]
#             sample= {"imgs":imgs,"labs":labs}
#             if self.transform:
#                 sample = self.transform(sample)
#             return sample
#     def __len__(self):
#         imgs,labs=self.data
#         return len(imgs)
with h5py.File('..\\dataset2\\train.h5', 'r') as file_object:
    rgb = file_object['rgb']  # images
    seg = file_object['seg']  # labels

    image = np.zeros((2975, 256, 256, 3))
    for i in range(2975):
        image[i] = cv.resize(rgb[i], (256, 256))
    image = image.astype(np.uint8)
    # label = np.array(seg)# specify the index
    label = np.zeros((2975, 256, 256))
    color_codes = np.array(file_object['color_codes'])
    for i in range(2975):
        label[i] = cv.resize(seg[i], (256, 256))
    label = label.astype(np.uint8)
    # image = np.transpose(image, (0, 3, 1, 2))
    print(image.shape)
    print(label.shape)


with h5py.File('..\\dataset2\\test.h5', 'r') as test_object:
    rgb2 = test_object['rgb']  # images
    seg2 = test_object['seg']  # labels
    image2 = np.zeros((500, 256, 256, 3))
    for i in range(500):
        image2[i] = cv.resize(rgb2[i], (256, 256))
    image2 = image2.astype(np.uint8)
    # label = np.array(seg)# specify the index
    label2 = np.zeros((500, 256, 256))
    for i in range(500):
        label2[i] = cv.resize(seg2[i], (256, 256))
    label2 = label2.astype(np.uint8)



class GetLoader(Dataset):
    def __init__(self, data_root, data_label, transform=None):
        self.data = data_root
        self.label = data_label
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        if self.transform is not None:
            # data = torch.from_numpy(data)
            data = self.transform(data)
            labels = torch.from_numpy(labels)
            # labels = self.transform(labels)
        sample = {"img": data, "lab": labels}
        return sample

    def __len__(self):
        return len(self.data)


class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(p=0.15)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(p=0.15)
        )

    def forward(self, x):
        out = self.Conv_BN_ReLU_2(x)
        out_2 = self.downsample(out)
        return out, out_2

class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(),
            nn.Dropout(p=0.15)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(p=0.15)
        )

    def forward(self, x, out):
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        cat_out = torch.cat((x_out, out), dim=1)
        return cat_out

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels = [2 ** (i + 6) for i in range(5)]  # [64, 128, 256, 512, 1024]
        # downsampling
        self.d1 = DownsampleLayer(3, out_channels[0])  # 3-64
        self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
        self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
        self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
        # upsampling
        self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
        self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
        self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
        self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
        # output
        self.o = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[0], kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Conv2d(out_channels[0], 34, 3, 1, 1),
            #nn.Sigmoid(),
            # BCELoss
        )

    def forward(self, x):
        out_1, out1 = self.d1(x)
        out_2, out2 = self.d2(out1)
        out_3, out3 = self.d3(out2)
        out_4, out4 = self.d4(out3)
        out5 = self.u1(out4, out_4)
        out6 = self.u2(out5, out_3)
        out7 = self.u3(out6, out_2)
        out8 = self.u4(out7, out_1)
        out = self.o(out8)
        return out

def train():
    seed_torch(1)
    Epoch = 8
    batch_size= 8
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # trainset= Cityscape(root_dir="..\\dataset2\\train.h5",transform=transform)
    train_data = GetLoader(image, label, transform=transform)
    # testset = Cityscape(root_dir="../dataset2/test.h5", transform= transform)
    dataloader= DataLoader(train_data,batch_size= batch_size, shuffle=True, num_workers=0,drop_last=True)
    criterion = nn.CrossEntropyLoss()
    net = UNet()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(Epoch):
        print("The %d th epoch begin" % (epoch+1))
        loss_list = []
        for i,data in enumerate(dataloader):
            imgs, labs = data["img"],data["lab"]
            imgs, labs = imgs.to(device), labs.to(device)
            output = net(imgs)
            print(output.shape)
            print(labs.shape)
            loss = criterion(output, labs.long())
            loss_list.append(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # summary.add_scalar('bceloss', loss, i)
            print("finished %d th iteration" % (i))
        loss= np.asarray(loss_list).mean()
        print("the loss is %.3f" % (loss))
        torch.save(net.state_dict(), './checkpoints/Epoch_{}.pt'.format(epoch+1))
        print('{}th epoch finished!'.format(epoch))

if __name__=="__main__":
    train()



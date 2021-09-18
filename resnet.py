import torch
import torch.nn as nn
import torch.nn.functional as F

# For the first block alone

class block0(nn.Module):
  def __init__(self):
    super(block0, self).__init__()
    
    self.conv = nn.Conv2d(64,64,kernel_size=3,stride=1,padding='same',bias=False)
    self.bn = nn.BatchNorm2d(64)

  def forward(self, X):
    I = X
    X = self.conv(X)
    X = self.bn(X)
    X = self.bn(self.conv(X))
    X += I
    I = X
    X = self.bn(self.conv(X))
    X = F.relu(X)
    X = self.bn(self.conv(X))

    return X

# For rest of the blocks (general implementation)

class block(nn.Module):
  def __init__(self, nc_in, nc_out):
    super(block,self).__init__()

    self.conv_stride2 = nn.Conv2d(nc_in,nc_out,kernel_size=3,stride=2,padding=1,bias=False)
    self.bn = nn.BatchNorm2d(nc_out)
    self.conv_stride1 = nn.Conv2d(nc_out,nc_out,kernel_size=3,stride=1,padding='same',bias=False)
    self.resize = nn.Conv2d(nc_in,nc_out,kernel_size=1,stride=2,bias=False)

  def forward(self,X):
    I = X
    X = self.conv_stride2(X)
    X = F.relu(self.bn(X))
    X = self.bn(self.conv_stride1(X))
    I = self.bn(self.resize(I))
    X += I
    I = X
    X = F.relu(self.bn(self.conv_stride1(X)))
    X = self.bn(self.conv_stride1(X))
    X += I

    return X

# Incorporating all the blocks

class ResNet18(nn.Module):
  def __init__(self):
    super(ResNet18, self).__init__()

    self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
    self.bn = nn.BatchNorm2d(64)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer0 = block0()
    self.layer1 = block(64,128)
    self.layer2 = block(128,256)
    self.layer3 = block(256,512)
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    self.fc1 = nn.Linear(512,1000)
    self.fc2 = nn.Linear(1000,5)

  def forward(self,X):
    X = self.conv1(X)
    X = self.bn(X)
    X = F.relu(X)
    X = self.maxpool(X)
    X = self.layer0(X)
    X = self.layer1(X)
    X = self.layer2(X)
    X = self.layer3(X)
    X = self.avgpool(X)
    X = torch.reshape(X,(-1,1,512))
    X = self.fc1(X)
    X = self.fc2(X)

    return X

model = ResNet18()

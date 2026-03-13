"""
meso-inception4
Architecture:
- 2 inception blocks with batch norm and max pooling
- 2 standard 2d convolutional layers with batchnorm/pool
- fully connected layers with dropout for classification

InceptionBlocks:
- ax(1x1) - 1x1 + ReLU
- bx(1x1) + ReLU -> bx(3x3): 1x1 bottleneck, then 3x3 conv
- cx(1x1) + ReLU -> dilated conv2, cx(3x3): 1x1 bottleneck then dilated 3x3 (dilation=2)
- dx(1x1) + ReLU -> dilated conv 3, dx(3x3): 1x1 bottleneck, then diilated 3x3 (dilation-3)

inception layer 1 params:  a=1, b=4, c=4, d=1
inception layer 2 params: a=1, b=4, c=4, d=2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, a, b, c, d): #a/b/c/d are the output filters
        super(InceptionBlock, self).__init__()

        #branch1: 1x1 conv only
        self.branch1 = nn.Conv2d(in_ch, a, kernel_size=1, padding=0,bias=False)

        #branch2: 1x1 bottleneck -> 3x3 conv
        self.branch2_1x1 = nn.Conv2d(in_ch, b, kernel_size=1, padding=0, bias=False)
        self.branch2_3x3 = nn.Conv2d(b, b, kernel_size=3, padding=1, bias=False)

        #branch3: 1x1 bottleneck -> dilated 3x3 (dilation=2)
        self.branch3_1x1 = nn.Conv2d(in_ch, c, kernel_size=1, padding=0,bias=False)
        self.branch3_3x3 = nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2, bias=False)

        #branch4: 1x1 bottleneck -> dilated 3x3 (dilation=3)
        self.branch4_1x1 = nn.Conv2d(in_ch, d, kernel_size=1, padding=0,bias=False)
        self.branch4_3x3 = nn.Conv2d(d, d, kernel_size=3, padding=3, dilation=3, bias=False)

        self.ReLU = nn.ReLU()
        self.leakyrely = nn.LeakyReLU(0.1)

    def forward(self, x):
        #branch 1: just 1x1
        branch1_out = self.ReLU(self.branch1(x))

        #branch 2: 1x1 -> ReLU -> 3x3
        branch2_out = self.ReLU(self.branch2_1x1(x))
        branch2_out = self.ReLU(self.branch2_3x3(branch2_out))

        #branch 3: 1x1 -> ReLU -> dilated 3x3 (dilation=2)
        branch3_out = self.ReLU(self.branch3_1x1(x))
        branch3_out = self.ReLU(self.branch3_3x3(branch3_out))

        #branch 4: 1x1 -> ReLU -> dilated 3x3 (dilation=3)
        branch4_out = self.ReLU(self.branch4_1x1(x))
        branch4_out = self.ReLU(self.branch4_3x3(branch4_out))

        #concat all branches
        output = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)
        return output
    
class MesoInception4(nn.Module):
    def __init__(self, num_classes=2):
        super(MesoInception4, self).__init__()

        #first inception block
        #input=256x256x3
        #using modern approach for incpetionblockk params: a=1, b=4, c=4, d=2
        #instead of a=1, b=4, c=4, d=1 -> 10 channels
        self.inception1 = InceptionBlock(3, a=1, b=4, c=4, d=2)  # output: 1+4+4+1 = 11 channels        
        self.inception1_bn = nn.BatchNorm2d(11)
        self.inception1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #second inception block
        #input = 128x128x10
        self.inception2 = InceptionBlock(11, a=2, b=4, c=4, d=2)  # output: 2+4+4+2 = 12 channels
        self.inception2_bn = nn.BatchNorm2d(12)
        self.inception2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #standard convolution layers
        #input 64x64x11
        self.conv1 = nn.Conv2d(12, 16, kernel_size=5, padding=2) #11->16 channels
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #input 32x32x16
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv2_pool = nn.MaxPool2d(kernel_size=4, stride=4)

        #classifier
        #input = 8x8x16
        self.fc1 = nn.Linear(1024, 16)  #16*8*8 = 1024
        self.fc2 = nn.Linear(16, 1)

        #ReLU and dropout
        self.leakyReLU = nn.LeakyReLU(0.1)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        #first inception block
        x = self.inception1(x)
        x = self.inception1_bn(x)    
        x = self.inception1_pool(x)

        #second inception block
        x = self.inception2(x)
        x = self.inception2_bn(x)    
        x = self.inception2_pool(x)

        #convolutional layers
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.ReLU(x)
        x = self.conv1_pool(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.ReLU(x)
        x = self.conv2_pool(x)

        #flatten for fully connected layers
        x = torch.flatten(x, start_dim=1)

        #fullyConnected layers
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.leakyReLU(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

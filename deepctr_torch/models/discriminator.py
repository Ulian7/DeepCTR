

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function



class Discriminator(nn.Module):
    def __init__(self, outputs_size, K = 2):
        super(Discriminator, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=outputs_size, out_channels=outputs_size//K, kernel_size=1, stride=1, bias=True)
        # outputs_size = outputs_size // K
        # self.conv2 = nn.Conv2d(in_channels=outputs_size, out_channels=outputs_size//K, kernel_size=1, stride=1, bias=True)
        # outputs_size = outputs_size // K
        # self.conv3 = nn.Conv2d(in_channels=outputs_size, out_channels=2, kernel_size=1, stride=1, bias=True)
        self.fc1 = nn.Linear(outputs_size, outputs_size//K, bias=True)
        outputs_size  = outputs_size // K
        self.fc2 = nn.Linear(outputs_size, outputs_size//K, bias=True)
        outputs_size  = outputs_size // K
        self.fc3 = nn.Linear(outputs_size, 2, bias=True)


    def forward(self, x):
        x = x[:,:,None,None]
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = out.view(out.size(0), -1)
        return out

class Discriminators(nn.Module):
    def __init__(self, output_dims):
        super(Discriminators, self).__init__()
        self.discriminators = [Discriminator(i) for i in output_dims]

    def forward(self, x):
        out = [self.discriminators[i](x[i]) for i in range(len(self.discriminators))]

        return out





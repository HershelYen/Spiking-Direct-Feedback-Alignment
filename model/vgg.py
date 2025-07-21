'''
creater: Hershel
since: 2024-11-04 00:39:47
LastAuthor: Hershel
lastTime: 2024-11-04 01:08:18
文件相对于项目的路径: /snn_myself/resnet/vgg.py
message: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from typing import Type, Any, Callable, Union, List, Optional

from spikingjelly.activation_based import neuron, functional, surrogate, layer
from module.sdfa import DFA, DFALayer
import torch.nn.functional as F
# the VGG11 architecture
class VGG11_SNN(nn.Module):
    def __init__(self,
                 data_set,
                 channels=512,
                 training_method='bp',
                 output_classes=10,
                 use_cupy=False,
                 spiking_neuron: callable = None, *args, **kwargs):
        super(VGG11_SNN, self).__init__()
        self.in_channels = 2
        self.output_classes = output_classes
        self.training_method = training_method
        self.conv1 = nn.Sequential(
            layer.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False),
            spiking_neuron(*args, **kwargs),
            layer.MaxPool2d(2, 2),#64
            #layer.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            layer.Conv2d(channels//8, channels//4, kernel_size=3, stride=1, padding=1, bias=False),
            spiking_neuron(*args, **kwargs),
            layer.MaxPool2d(2, 2),#32
            #layer.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            layer.Conv2d(channels//4, channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            spiking_neuron(*args, **kwargs),
            #layer.Dropout(0.5),
        )
        self.conv4 = nn.Sequential(
            layer.Conv2d(channels//2, channels//2, kernel_size=3, stride=1, padding=1, bias=False),
            spiking_neuron(*args, **kwargs),
            layer.MaxPool2d(2, 2),#16
            #layer.Dropout(0.5),
        )
        self.conv5 = nn.Sequential(
            layer.Conv2d(channels//2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            spiking_neuron(*args, **kwargs),
            #layer.Dropout(0.5),
        )
        self.conv6 = nn.Sequential(
            layer.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            spiking_neuron(*args, **kwargs),
            layer.MaxPool2d(2, 2),#8
            #layer.Dropout(0.5),
        )
        self.conv7 = nn.Sequential(
            layer.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            spiking_neuron(*args, **kwargs),
            #layer.Dropout(0.5),
        )
        self.conv8 = nn.Sequential(
            layer.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            spiking_neuron(*args, **kwargs),
            layer.MaxPool2d(2, 2),#4
            
        )
        if data_set == 'dvs':
            self.fc1 = nn.Sequential(
                layer.Flatten(),
                #layer.Dropout(0.5),
                layer.Linear(in_features=channels*4*4, out_features=4096),
                spiking_neuron(*args, **kwargs),
            )
        elif data_set == 'ncaltech':
            self.fc1 = nn.Sequential(
                layer.Flatten(),
                #layer.Dropout(0.5),
                layer.Linear(in_features=channels*5*7, out_features=4096),
                spiking_neuron(*args, **kwargs),
            )
        self.fc2 = nn.Sequential(
            layer.Linear(in_features=4096, out_features=4096),
            spiking_neuron(*args, **kwargs),
        )
        self.fc3 = nn.Sequential(
            #layer.Dropout(0.5),
            layer.Linear(in_features=4096, out_features=output_classes*10),
            spiking_neuron(*args, **kwargs),
            layer.VotingLayer(10)
        )
        if self.training_method in ['sdfa', 'shallow']:
            self.dfa1= DFALayer(batch_dims=(1, ), time_dims=(0, ))
            self.dfa2= DFALayer(batch_dims=(1, ), time_dims=(0, ))
            self.dfa3= DFALayer(batch_dims=(1, ), time_dims=(0, ))
            self.dfa4= DFALayer(batch_dims=(1, ), time_dims=(0, ))
            self.dfa5= DFALayer(batch_dims=(1, ), time_dims=(0, ))
            self.dfa6= DFALayer(batch_dims=(1, ), time_dims=(0, ))
            self.dfa7= DFALayer(batch_dims=(1, ), time_dims=(0, ))
            self.dfa8= DFALayer(batch_dims=(1, ), time_dims=(0, ))
            self.dfa9= DFALayer(batch_dims=(1, ), time_dims=(0, ))
            self.dfa10= DFALayer(batch_dims=(1, ), time_dims=(0, ))
            dfa_list = [self.dfa1, self.dfa2, self.dfa3, self.dfa4, self.dfa5, self.dfa6, self.dfa7, self.dfa8, self.dfa9, self.dfa10]
            self.dfa = DFA(dfa_list, no_training=(self.training_method=='shallow'))
            print(dfa_list)
        functional.set_step_mode(self, step_mode='m')

        if use_cupy:
            functional.set_backend(self, backend='cupy')
    def forward(self, x):
        if self.training_method in ['sdfa', 'shallow']:
            x = self.dfa1(self.conv1(x))
            x = self.dfa2(self.conv2(x))
            x = self.dfa3(self.conv3(x))
            x = self.dfa4(self.conv4(x))
            x = self.dfa5(self.conv5(x))
            x = self.dfa6(self.conv6(x))
            x = self.dfa7(self.conv7(x))
            x = self.dfa8(self.conv8(x))
            x = self.dfa9(self.fc1(x))
            x = self.dfa10(self.fc2(x))
            x = self.dfa(self.fc3(x).mean(0))
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.conv8(x)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x).mean(0)
        return x

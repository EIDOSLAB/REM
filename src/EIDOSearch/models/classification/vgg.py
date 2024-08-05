#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

from torch import nn


# TODO this sucks!
class CifarVGG:
    def __init__(self, fc_layers, classes):
        self.cfg = {'VGG11': [64, 'M',
                              128, 'M',
                              256, 256, 'M',
                              512, 512, 'M',
                              512, 512, 'M'],
                    'VGG13': [64, 64, 'M',
                              128, 128, 'M',
                              256, 256, 'M',
                              512, 512, 'M',
                              512, 512, 'M'],
                    'VGG16': [64, 64, 'M',
                              128, 128, 'M',
                              256, 256, 256, 'M',
                              512, 512, 512, 'M',
                              512, 512, 512, 'M'],
                    'VGG19': [64, 64, 'M',
                              128, 128, 'M',
                              256, 256, 256, 256, 'M',
                              512, 512, 512, 512, 'M',
                              512, 512, 512, 512, 'M']}
        if fc_layers == 1:
            self.vgg = CifarVGG.VGG1L(classes, self.cfg, "VGG16")
        elif fc_layers == 2:
            self.vgg = CifarVGG.VGG2L(classes)
        else:
            raise ValueError(f"Incorrect argument, fc_layers must be either 1 or 2. Found {fc_layers}.")
    
    def __call__(self):
        return self.vgg
    
    class VGG1L(nn.Module):
        def __init__(self, name, cfg, classes):
            super(CifarVGG.VGG1L, self).__init__()
            self.features = self._make_layers(cfg[name])
            self.classifier = nn.Linear(512, classes)
        
        def forward(self, x):
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out
        
        @staticmethod
        def _make_layers(cfg):
            layers = []
            in_channels = 3
            for x in cfg:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                    in_channels = x
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
            return nn.Sequential(*layers)
    
    class VGG2L(nn.Module):
        def __init__(self, classes):
            super(CifarVGG.VGG2L, self).__init__()
            self.features = self._make_layers()
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
        
        @staticmethod
        def _make_layers():
            layers = []
            layers += [nn.Conv2d(3, 64, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(64, eps=1e-3)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout(0.3)]
            
            layers += [nn.Conv2d(64, 64, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(64, eps=1e-3)]
            layers += [nn.ReLU()]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            
            layers += [nn.Conv2d(64, 128, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(128, eps=1e-3)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout(0.4)]
            
            layers += [nn.Conv2d(128, 128, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(128, eps=1e-3)]
            layers += [nn.ReLU()]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            
            layers += [nn.Conv2d(128, 256, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(256, eps=1e-3)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout(0.4)]
            
            layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(256, eps=1e-3)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout(0.4)]
            
            layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(256, eps=1e-3)]
            layers += [nn.ReLU()]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            
            layers += [nn.Conv2d(256, 512, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(512, eps=1e-3)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout(0.4)]
            
            layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(512, eps=1e-3)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout(0.4)]
            
            layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(512, eps=1e-3)]
            layers += [nn.ReLU()]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            
            layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(512, eps=1e-3)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout(0.4)]
            
            layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(512, eps=1e-3)]
            layers += [nn.ReLU()]
            layers += [nn.Dropout(0.4)]
            
            layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
            layers += [nn.BatchNorm2d(512, eps=1e-3)]
            layers += [nn.ReLU()]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            return nn.Sequential(*layers)

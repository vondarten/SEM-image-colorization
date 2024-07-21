import torch
import torch.nn as nn

class PatchGAN(nn.Module):
    def __init__(self, in_channels=3, custom_weights_init=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=64,
                      kernel_size=4,  
                      stride=2,
                      padding=1,
                      bias=False
                      ),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=4,  
                      stride=2,
                      padding=1,
                      bias=False
                      ),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=4,  
                      stride=2,
                      padding=1,
                      bias=False
                      ),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=256),

            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=4,  
                      stride=1,
                      padding=1,
                      bias=False
                      ),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=512),

            nn.Conv2d(in_channels=512,
                      out_channels=1,
                      kernel_size=4,  
                      stride=1,
                      padding=1,
                      bias=False
                      )
        )

        if custom_weights_init: 
            self.model.apply(self._weights_initialization)
    
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(x)    
    
    def _weights_initialization(self, module, mean=0.0, std=0.02):
        if isinstance(module, torch.nn.Conv2d):
            nn.init.normal_(module.weight.data, mean=mean, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0.0)

        elif isinstance(module, torch.nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(module.bias.data, 0.0)
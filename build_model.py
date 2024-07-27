import torch
import torchvision.models as models
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet


def build_model(backbone: str, 
                n_input: int, 
                n_output: int, 
                size: int, 
                activation_function: str, 
                self_attention: bool
                ) -> DynamicUnet:

    if backbone == 'resnet18':
        backbone = models.resnet18(weights='DEFAULT')
    
    elif backbone == 'resnet34':
        backbone = models.resnet34(weights='DEFAULT')
    
    elif backbone == 'shufflenet_v2_x0_5':
        backbone = models.shufflenet_v2_x2_0(weights='DEFAULT')

    elif backbone == 'shufflenet_v2_x1_0':
        backbone = models.shufflenet_v2_x2_0(weights='DEFAULT')

    elif backbone == 'shufflenet_v2_x1_5':
        backbone = models.shufflenet_v2_x2_0(weights='DEFAULT')

    elif backbone == 'shufflenet_v2_x2_0':
        backbone = models.shufflenet_v2_x2_0(weights='DEFAULT')
  
    body = create_body(backbone, n_in=n_input, pretrained=True, cut=-2) 

    if activation_function:
        return DynamicUnet(body, n_output, (size, size), self_attention=self_attention, act_cls=torch.nn.Mish)

    return DynamicUnet(body, n_output, (size, size), self_attention=self_attention)
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.models as models
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet


DINOV3_REPO = 'facebookresearch/dinov3'
DINOV3_CHECKPOINT_DIR = Path(__file__).resolve().parent / 'checkpoints'


TORCHVISION_BACKBONES = {
    'convnext_tiny': models.convnext_tiny,
    'convnext_small': models.convnext_small,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b1': models.efficientnet_b1,
    'efficientnet_b2': models.efficientnet_b2,
    'efficientnet_v2_s': models.efficientnet_v2_s,
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    'mobilenet_v3_small': models.mobilenet_v3_small,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnext101_32x8d': models.resnext101_32x8d,
    'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': models.shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': models.shufflenet_v2_x2_0,
    'squeezenet1_0': models.squeezenet1_0,
    'squeezenet1_1': models.squeezenet1_1,
    'wide_resnet50_2': models.wide_resnet50_2,
    'wide_resnet101_2': models.wide_resnet101_2,
}

DINOv3_BACKBONES = {
    'dinov3_convnext_tiny': 'dinov3_convnext_tiny',
}

DINOv3_LOCAL_WEIGHTS = {
    'dinov3_convnext_tiny': 'dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
}


def _load_backbone(backbone: str) -> torch.nn.Module:
    if backbone in TORCHVISION_BACKBONES:
        return TORCHVISION_BACKBONES[backbone](weights='DEFAULT')

    if backbone in DINOv3_BACKBONES:
        return _load_dinov3_backbone(DINOv3_BACKBONES[backbone])

    supported_backbones = sorted(TORCHVISION_BACKBONES | DINOv3_BACKBONES)
    raise ValueError(f'Unsupported backbone {backbone}. Supported backbones: {supported_backbones}')


def _load_dinov3_backbone(model_name: str) -> torch.nn.Module:
    pretrained = os.getenv('DINOV3_PRETRAINED', '1') not in {'0', 'false', 'False'}
    weights = os.getenv('DINOV3_WEIGHTS') or _get_local_dinov3_weights(model_name)

    try:
        hub_kwargs = {
            'repo_or_dir': DINOV3_REPO,
            'model': model_name,
            'pretrained': pretrained,
            'trust_repo': True,
        }
        if weights:
            hub_kwargs['weights'] = weights

        return torch.hub.load(
            **hub_kwargs,
        )
    except (ImportError, OSError) as error:
        if isinstance(error, ImportError) and 'custom_fwd' not in str(error):
            raise
        if not isinstance(error, ImportError) and pretrained:
            raise RuntimeError(
                'Could not load DINOv3 pretrained weights. Set DINOV3_WEIGHTS to a local '
                'checkpoint/URL, or set DINOV3_PRETRAINED=0 to train from random initialization.'
            ) from error

        repo_dir = torch.hub._get_cache_or_reload(
            github=DINOV3_REPO,
            force_reload=False,
            trust_repo=True,
            calling_fn='_load_dinov3_backbone',
            verbose=True,
            skip_validation=False,
        )
        return _load_dinov3_backbone_from_repo(Path(repo_dir), model_name, pretrained, weights)


def _get_local_dinov3_weights(model_name: str) -> str | None:
    weights_name = DINOv3_LOCAL_WEIGHTS.get(model_name)
    if weights_name is None:
        return None

    weights_path = DINOV3_CHECKPOINT_DIR / weights_name
    if weights_path.exists():
        return str(weights_path)

    return None


def _load_dinov3_backbone_from_repo(
    repo_dir: Path,
    model_name: str,
    pretrained: bool,
    weights: str | None,
) -> torch.nn.Module:
    sys.path.insert(0, str(repo_dir))
    try:
        from dinov3.hub import backbones

        kwargs = {'pretrained': pretrained}
        if weights:
            kwargs['weights'] = weights

        return getattr(backbones, model_name)(**kwargs)
    except OSError as error:
        if pretrained:
            raise RuntimeError(
                'Could not load DINOv3 pretrained weights. Set DINOV3_WEIGHTS to a local '
                'checkpoint/URL, or set DINOV3_PRETRAINED=0 to train from random initialization.'
            ) from error
        raise
    finally:
        sys.path.remove(str(repo_dir))


def _build_dinov3_convnext_body(backbone: torch.nn.Module, n_input: int) -> torch.nn.Sequential:
    _adapt_first_conv(backbone, n_input)

    layers = []
    for downsample_layer, stage in zip(backbone.downsample_layers, backbone.stages):
        layers.extend([downsample_layer, stage])

    return torch.nn.Sequential(*layers)


def _adapt_first_conv(backbone: torch.nn.Module, n_input: int) -> None:
    first_conv = backbone.downsample_layers[0][0]
    if first_conv.in_channels == n_input:
        return

    new_conv = torch.nn.Conv2d(
        in_channels=n_input,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode,
    ).to(device=first_conv.weight.device, dtype=first_conv.weight.dtype)

    with torch.no_grad():
        averaged_weights = first_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.copy_(averaged_weights.repeat(1, n_input, 1, 1))
        if n_input > 1:
            new_conv.weight.mul_(first_conv.in_channels / n_input)
        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)

    backbone.downsample_layers[0][0] = new_conv


class DINOv3ColorizationModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, n_input: int, n_output: int):
        super().__init__()
        _adapt_first_conv(backbone, n_input)
        self.backbone = backbone
        self.output_head = torch.nn.Conv2d(768, n_output, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)['x_norm_patchtokens']
        batch_size, n_tokens, n_channels = features.shape
        feature_size = int(n_tokens ** 0.5)
        features = features.transpose(1, 2).reshape(batch_size, n_channels, feature_size, feature_size)
        output = self.output_head(features)
        return F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=False)


def build_model(backbone: str, 
                n_input: int, 
                n_output: int, 
                size: int, 
                activation_function: str, 
                self_attention: bool
                ) -> torch.nn.Module:

    backbone_name = backbone
    backbone = _load_backbone(backbone_name)
  
    if backbone_name in DINOv3_BACKBONES:
        return DINOv3ColorizationModel(backbone, n_input, n_output)
    else:
        body = create_body(backbone, n_in=n_input, pretrained=True, cut=-2)

    if activation_function:
        return DynamicUnet(body, n_output, (size, size), self_attention=self_attention, act_cls=torch.nn.Mish)

    return DynamicUnet(body, n_output, (size, size), self_attention=self_attention)
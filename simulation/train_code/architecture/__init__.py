import torch
from .CFSDCN import CFSDCN

def model_generator(method, pretrained_model_path=None):
    if method == 'cfsdcn_s':
        model = CFSDCN(dim=28, stage=2, num_blocks=[1, 2, 2], groups=7).cuda()
    elif method == 'cfsdcn_m':
        model = CFSDCN(dim=28, stage=2, num_blocks=[2, 4, 6], groups=7).cuda()
    elif method == 'cfsdcn_l':
        model = CFSDCN(dim=28, stage=2, num_blocks=[4, 8, 10], groups=7).cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()},
                              strict=False)
    return model
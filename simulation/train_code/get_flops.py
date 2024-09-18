import copy

import torch
from architecture.CFSDCN import CFSDCN


from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string
from mmcv.cnn import get_model_complexity_info

def dcnv3_flops(n, k, c):
    return 5 * n * k * c


def get_flops(model, input_shape, num_blocks=None):
    if num_blocks is None:
        num_blocks = [1, 2, 2]
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False)

    channels, H, W = input_shape

    temp = 0
    depths_down = [num_blocks[0],num_blocks[1]]
    depths_bottleneck = [num_blocks[2]]
    depths_up = [num_blocks[0],num_blocks[1]]
    # depth = 1
    h = H
    w = W
    for idx, depth in enumerate(depths_down):
        temp += depth * dcnv3_flops(n=h * w, k=3 * 3, c=channels)
        h = h / 2
        w = w / 2
        channels *= 2
    for idx, depth in enumerate(depths_bottleneck):
        temp += depth * dcnv3_flops(n=h * w, k=3 * 3, c=channels)
    for idx, depth in enumerate(depths_up):
        temp += depth * dcnv3_flops(n=h * w, k=3 * 3, c=channels)
        h = h * 2
        w = w * 2
        channels /= 2
    flops = flops + temp
    return flops_to_string(flops), params_to_string(params)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    orig_shape = (28, 256, 256)

    input_shape = (28, 256, 256)
    num_blocks = [4, 8, 8]

    modelCFSDCN = CFSDCN(dim=28, stage=2, num_blocks=num_blocks, groups=7).cuda()

    modelCFSDCN.eval()
    modelCFSDCN.cuda()

    flops, params = get_flops(modelCFSDCN, input_shape, num_blocks=num_blocks)
    split_line = '=' * 30

    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
from timm.models.efficientnet_builder import *
from timm.models.efficientnet import EfficientNet, EfficientNetFeatures, _cfg
from timm.models.registry import register_model


def _gen_tinynet(variant_cfg, channel_multiplier=1.0, depth_multiplier=1.0, depth_trunc='round', **kwargs):
    """Creates a TinyNet model.
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'], ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'], ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'], ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc=depth_trunc),
        num_features=max(1280, round_channels(1280, channel_multiplier, 8, None)),
        stem_size=32,
        fix_stem=True,
        channel_multiplier=channel_multiplier,
        act_layer=Swish,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs,
    ) 
    model = EfficientNet(**model_kwargs)
    model.default_cfg = variant_cfg
    return model


@register_model
def tinynet(r=1.0, w=1.0, d=1.0, **kwargs):
    """ TinyNet """
    hw = int(224 * r)
    model = _gen_tinynet(
        _cfg(input_size=(3, hw, hw)), channel_multiplier=w, depth_multiplier=d, **kwargs)
    return model


import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.parallel


from timm.models import create_model, load_checkpoint, is_model
from timm.data import Dataset, create_loader, resolve_data_config
from timm.utils import accuracy, AverageMeter, setup_default_logging

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model_name', default='tinynet-a',
                    help='model architecture (default: tinynet-a)')


def validate(args):
    # create model
    if args.model_name == 'tinynet_a':
        args.r=0.86
        args.w=1.0
        args.d=1.2
        ckpt_path = './models/tinynet_a.pth'
    elif args.model_name == 'tinynet_b':
        args.r=0.84
        args.w=0.75
        args.d=1.1
        ckpt_path = './models/tinynet_b.pth'
    elif args.model_name == 'tinynet_c':
        args.r=0.825
        args.w=0.54
        args.d=0.85
        ckpt_path = './models/tinynet_c.pth'
    elif args.model_name == 'tinynet_d':
        args.r=0.68
        args.w=0.54
        args.d=0.695
        ckpt_path = './models/tinynet_d.pth'
    elif args.model_name == 'tinynet_e':
        args.r=0.475
        args.w=0.51
        args.d=0.60
        ckpt_path = './models/tinynet_e.pth'
    else:
        raise 'Unsupported model name.'

    model = tinynet(
        r=args.r,
        w=args.w,
        d=args.d,)

    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict, strict=False)

    params = sum([param.numel() for param in model.parameters()])
    logging.info('Model %s created, #params: %d' % (args.model_name, params))

    data_config = resolve_data_config(vars(args), model=model)

    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    
    dataset = Dataset(args.data)
    data_loader = create_loader(
        dataset,
        is_training=False,
        input_size=data_config['input_size'],
        batch_size=128,
        use_prefetcher=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=4,
        crop_pct=data_config['crop_pct'],
        pin_memory=False)

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            if i % 100 == 0:
                logging.info(
                    'Test: [{0:>4d}/{1}]  Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})'.format(
                        i, len(data_loader), loss=losses))
    
    logging.info(' * Acc@1 {:.3f} Acc@5 {:.3f}'.format(top1.avg, top5.avg))


def main():
    setup_default_logging()
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()
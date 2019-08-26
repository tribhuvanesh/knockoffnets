import torch
import torch.nn as nn
import os.path as osp

import knockoff.models.cifar
import knockoff.models.mnist
import knockoff.models.imagenet


def get_net(modelname, modeltype, pretrained=None, **kwargs):
    assert modeltype in ('mnist', 'cifar', 'imagenet')
    if pretrained and pretrained is not None:
        return get_pretrainednet(modelname, modeltype, pretrained, **kwargs)
    else:
        try:
            # This should have ideally worked:
            model = eval('knockoff.models.{}.{}'.format(modeltype, modelname))(**kwargs)
        except AssertionError:
            # But, there's a bug in pretrained models which ignores the num_classes attribute.
            # So, temporarily load the model and replace the last linear layer
            model = eval('knockoff.models.{}.{}'.format(modeltype, modelname))()
            if 'num_classes' in kwargs:
                num_classes = kwargs['num_classes']
                in_feat = model.last_linear.in_features
                model.last_linear = nn.Linear(in_feat, num_classes)
        return model


def get_pretrainednet(modelname, modeltype, pretrained='imagenet', num_classes=1000, **kwargs):
    if pretrained == 'imagenet':
        return get_imagenet_pretrainednet(modelname, num_classes, **kwargs)
    elif osp.exists(pretrained):
        model = eval('knockoff.models.{}.{}'.format(modeltype, modelname))(num_classes=num_classes, **kwargs)
        checkpoint = torch.load(pretrained)
        pretrained_state_dict = checkpoint.get('state_dict', checkpoint)
        copy_weights_(pretrained_state_dict, model.state_dict())
        return model
    else:
        raise ValueError('Currently only supported for imagenet or existing pretrained models')


def get_imagenet_pretrainednet(modelname, num_classes=1000, **kwargs):
    valid_models = knockoff.models.imagenet.__dict__.keys()
    assert modelname in valid_models, 'Model not recognized, Supported models = {}'.format(valid_models)
    model = knockoff.models.imagenet.__dict__[modelname](pretrained='imagenet')
    if num_classes != 1000:
        # Replace last linear layer
        in_features = model.last_linear.in_features
        out_features = num_classes
        model.last_linear = nn.Linear(in_features, out_features, bias=True)
    return model


def copy_weights_(src_state_dict, dst_state_dict):
    n_params = len(src_state_dict)
    n_success, n_skipped, n_shape_mismatch = 0, 0, 0

    for i, (src_param_name, src_param) in enumerate(src_state_dict.items()):
        if src_param_name in dst_state_dict:
            dst_param = dst_state_dict[src_param_name]
            if dst_param.data.shape == src_param.data.shape:
                dst_param.data.copy_(src_param.data)
                n_success += 1
            else:
                # print('Mismatch: {} ({} != {})'.format(src_param_name, dst_param.data.shape, src_param.data.shape))
                n_shape_mismatch += 1
        else:
            n_skipped += 1
    print('=> # Success param blocks loaded = {}/{}, '
          '# Skipped = {}, # Shape-mismatch = {}'.format(n_success, n_params, n_skipped, n_shape_mismatch))

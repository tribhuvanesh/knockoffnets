import torch.nn as nn

import knockoff.models.cifar
import knockoff.models.mnist
import knockoff.models.imagenet


def get_net(modelname, modeltype, pretrained=None, **kwargs):
    assert modeltype in ('mnist', 'cifar', 'imagenet')
    if pretrained:
        assert modeltype == 'imagenet', 'Currently only supported for pretrained models'
        return get_pretrainednet(modelname, **kwargs)
    else:
        return eval('knockoff.models.{}.{}'.format(modeltype, modelname))(**kwargs)


def get_pretrainednet(modelname, num_classes=1000, **kwargs):
    valid_models = knockoff.models.imagenet.__dict__.keys()
    assert modelname in valid_models, 'Model not recognized, Supported models = {}'.format(valid_models)

    model = knockoff.models.imagenet.__dict__[modelname](pretrained='imagenet')
    if num_classes != 1000:
        # Replace last linear layer
        in_features = model.last_linear.in_features
        out_features = num_classes
        model.last_linear = nn.Linear(in_features, out_features, bias=True)
    return model

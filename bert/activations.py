import logging
import math

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


ACT2FN = {
    "relu": F.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(activation_string, list(ACT2FN.keys())))

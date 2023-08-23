import sys

from .my_cfg import *
from .my_dataset import *


def get_config(parser):
    return getattr(sys.modules[__name__], f'get_config_{parser.DATASET}')(parser)

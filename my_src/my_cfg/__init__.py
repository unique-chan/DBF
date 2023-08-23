import sys

from .parser import *
from .my_cfg_for_amod_v1 import *


def get_config(parse_args):
    return getattr(sys.modules[__name__], f'get_config_{parse_args.DATASET}')(parse_args)

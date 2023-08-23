import importlib

from .parser import *


def get_config(parse_args):
    current_module = importlib.import_module(parse_args.CONFIG_FILE.replace("/", ".").replace(".py", ""))
    return current_module.get_config(parse_args, verbose=False)
    # return getattr(sys.modules[__name__], f'get_config_{parse_args.DATASET}')(parse_args)

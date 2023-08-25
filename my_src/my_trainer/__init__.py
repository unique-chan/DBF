import importlib

from .train import *


def _object_to_txt(txt, txt_file_path):
    with open(txt_file_path, 'w') as fp:
        if type(txt) is not str:
            txt = str(txt)
        fp.write(txt)


def save_log_from_runner(work_dir, runner):
    _object_to_txt(runner.meta, f'{work_dir}/runner.meta.txt')
    _object_to_txt(runner.outputs, f'{work_dir}/runner.outputs.txt')
    _object_to_txt('', f'{work_dir}/best_mAP_val_{runner.meta["hook_msgs"]["best_score"]}')


def get_DBF_scheduler_config(parse_args):
    current_module = importlib.import_module(parse_args.CONFIG_FILE.replace("/", ".").replace(".py", ""))
    return current_module.get_DBF_scheduler_config(parse_args)

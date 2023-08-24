from .train import *


def _object_to_txt(txt, txt_file_path):
    with open(txt_file_path, 'w') as fp:
        if type(txt) is not str:
            txt = str(txt)
        fp.write(txt)


def save_log_from_runner(work_dir, runner):
    # print(runner.meta)
    # print(runner.meta['hook_msgs']['best_score'])
    # print(runner.mode)
    # print(runner.model)
    # print(runner.model_name)
    # print(runner.optimizer)
    # print(runner.outputs)
    # print(runner.save_checkpoint)
    # print(runner.timestamp)
    _object_to_txt(runner.meta, f'{work_dir}/runner.meta.txt')
    _object_to_txt(runner.outputs, f'{work_dir}/runner.outputs.txt')
    _object_to_txt('', f'{work_dir}/best_mAP_val_{runner.meta["hook_msgs"]["best_score"]}')

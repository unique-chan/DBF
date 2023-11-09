import argparse

from mmcv import DictAction


class Parser:
    def __init__(self, mode):
        assert mode in ['train', 'eval'], f'Unsupported mode: {mode} for Parser'
        self.parser = argparse.ArgumentParser()
        self.add_common_arguments()
        if mode == 'train':
            self.add_train_arguments()
            self.add_DBF_arguments()        # for dynamic backbone freezing
        elif mode == 'eval':
            self.add_eval_arguments()

    def add_common_arguments(self):
        self.parser.add_argument('--model-config',
                                 help='model config file path, '
                                      'e.g. "mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"')
        self.parser.add_argument('--data-config',
                                 help='user-customized data-relevant config file path')
        self.parser.add_argument('--data-root', default='.', help='data root path')
        self.parser.add_argument('--work-dir', help='dir to save file containing eval metrics')
        self.parser.add_argument('--device', default='cuda',
                                 type=str, choices=['cpu', 'cuda'], help='"cpu" or "cuda"? (default: "cuda")')
        self.parser.add_argument('--gpu-id', type=int, default=0, nargs='+', help='id(s) of gpu(s) to use')

    def add_train_arguments(self):
        self.parser.add_argument('--train-config',
                                 help='user-customized training-relevant config file path')
        self.parser.add_argument('--epochs', type=int, help='training_epochs')
        self.parser.add_argument('--load-from', help='checkpoint file (weights only)')
        self.parser.add_argument('--resume-from', help='checkpoint file to resume from')
        self.parser.add_argument('--no-validate', action='store_true',
                                 help='whether not to evaluate the checkpoint during training')
        self.parser.add_argument('--seed', default=0, type=int, help='random seed (default: 0)')
        self.parser.add_argument('--deterministic', action='store_true',
                                 help='whether to set deterministic options for CUDNN backend')
        self.parser.add_argument('--init_weights', action='store_true',
                                 help='use init weights')
        self.parser.add_argument('--tag', help='experiment tag')

    def add_eval_arguments(self):
        self.parser.add_argument('--checkpoint', help='checkpoint file')
        self.parser.add_argument('--out', help='output result file in pickle format')
        self.parser.add_argument('--fuse-conv-bn', action='store_true',
                                 help='whether to fuse conv and bn, this will slightly increase the inference speed')
        self.parser.add_argument('--eval', type=str, nargs='+',
                                 help='eval metrics e.g. "bbox", "segm", "proposal" for COCO, and '
                                      '"mAP", "recall" for PASCAL VOC')
        self.parser.add_argument('--show', action='store_true', help='show results')
        self.parser.add_argument('--show-dir', help='dir where painted images will be saved')
        self.parser.add_argument('--show-score-thr', type=float, default=0.3,
                                 help='score threshold (default: 0.3)')
        self.parser.add_argument('--eval-options', nargs='+',
                                 action=DictAction,
                                 help='custom options for evaluation, the key-value pair in xxx=yyy '
                                      'format will be kwargs for dataset.evaluate() function')
        self.parser.add_argument('--format-only', action='store_true',
                                 help='Format the output results without perform evaluation. It is '
                                      'useful when you want to format the result to a specific format and '
                                      'submit it to the test server '
                                      '(check implemented format_results() for your mmdet.datasets.custom type)')

    def add_DBF_arguments(self):
        self.parser.add_argument('--dbf', help='file for dynamic backbone freezing')
        self.parser.add_argument('--dbf-options',
                                 help='scheduling options for dynamic backbone freezing '
                                      '▶ e.g. %s' % '{"step_epoch": 10}')

    def parse_args(self):
        return self.parser.parse_args()


# # zzz.py
# import xxx
#
# # cfg 변수를 만들고 xxx.py의 모든 변수를 자동으로 등록
# cfg = {}
# for var_name in dir(xxx):
#     if not var_name.startswith("__"):  # 필요 없는 변수(이중 밑줄로 시작하는)를 걸러냅니다.
#         setattr(cfg, var_name, getattr(xxx, var_name))
#
# # 이제 cfg.a 또는 cfg.b를 사용할 수 있습니다.
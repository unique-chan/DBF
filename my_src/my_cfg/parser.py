import argparse


def parse_range(input_str):
    # e.g. input_str: "(2,5)" -> return: range(2,5)
    try:
        parsed_list = eval(input_str)
        if not isinstance(parsed_list, (list, tuple)):
            raise argparse.ArgumentTypeError("Input is neither a list nor a tuple")
        return range(*parsed_list)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Error parsing input: {e}")


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='github.com/unique-chan/DBF')
        self.add_default_arguments()

    def add_default_arguments(self):
        self.parser.add_argument('--CONFIG_FILE', required=True, type=str,
                                 help='config file path ▶️ e.g. mmdetection/configs/yolox/***.py')
        self.parser.add_argument('--SAMPLES_PER_GPU', required=True, type=int,
                                 help='number of samples per gpu')
        self.parser.add_argument('--EPOCHS', required=True, type=int,
                                 help='epochs to train')
        self.parser.add_argument('--GPU_IDS', required=True, type=parse_range,
                                 help='gpu_ids          ▶️ e.g. "(0,3)"'
                                      '(hint) if "(0,3)" given, it will be interpreted as range(1,3), '
                                      '       which indicates that 0,1,2 gpus will be used.')
        self.parser.add_argument('--DATASET', required=True, type=str,
                                 help='dataset type     ▶️ e.g. "amod_v1"'
                                      '(hint) choose which class file (*.py) to import in my_dataset folder')
        self.parser.add_argument('--TAG_NAME', required=True, type=str,
                                 help='tag name for the current experiments')
        self.parser.add_argument('--LOAD_FROM', type=str,
                                 help='load_from        ▶️ e.g. checkpoints/yolox_***.pth'
                                      '(hint) compared to `resume_from`, this option only loads the model weights and'
                                      '       the training epoch starts from 0.')
        self.parser.add_argument('--RESUME_FROM', type=str,
                                 help='resume_from      ▶️ e.g. checkpoints/yolox_**epochs_***.pth'
                                      '(hint) compared to `load_from`, this option loads both the model weights and'
                                      '       optimizer status, and the epoch is also inherited from '
                                      '       the specified checkpoint.')
        self.parser.add_argument('--DEVICE', default='cuda', choices=['cpu', 'cuda'], type=str,
                                 help='"cpu" or "cuda"? (default: "cuda")')
        self.parser.add_argument('--SEED', default=0, type=int,
                                 help='random seed (default: 0)')

    def parse_args(self):
        return self.parser.parse_args()

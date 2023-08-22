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
        self.parser.add_argument('--config_file', type=str,
                                 help='config file path ▶️ e.g. mmdetection/configs/yolox/***.py')
        self.parser.add_argument('--samples_per_gpu', required=True, type=int,
                                 help='# samples per gpu')
        self.parser.add_argument('--epochs', required=True, type=int,
                                 help='epochs to train')
        self.parser.add_argument('--gpu_ids', required=True, type=parse_range,
                                 help='gpu_ids          ▶️ e.g. "(1,3)" -> range(1,3) -> gpu id 1,2 will be used')
        self.parser.add_argument('--tag_name', required=True, type=str,
                                 help='tag name for the current experiments')
        self.parser.add_argument('--load_from', type=str,
                                 help='load_from        ▶️ e.g. checkpoints/yolox_***.pth')
        self.parser.add_argument('--resume_from', type=str,
                                 help='resume_from      ▶️ e.g. checkpoints/yolox_**epochs_***.pth')
        self.parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], type=str,
                                 help='"cpu" or "cuda"? (default: "cuda")')
        self.parser.add_argument('--seed', default=0, type=int,
                                 help='random seed (default: 0)')

    def parse_args(self):
        return self.parser.parse_args()

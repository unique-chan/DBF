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
                                 help='path of config file')
        self.parser.add_argument('--samples_per_gpu', required=True, type=int,
                                 help='# samples per gpu')
        self.parser.add_argument('--epochs', required=True, type=int,
                                 help='epochs to train')
        self.parser.add_argument('--gpu_ids', required=True, type=parse_range,
                                 help='gpu_ids e.g. "(1,3)" -> range(1,3) -> gpu id 1,2 will be used')
        self.parser.add_argument('--tag_name', required=True, type=str,
                                 help='tag name for the current experiments')
        self.parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], type=str,
                                 help='"cpu" or "cuda"? (default: "cuda")')
        self.parser.add_argument('--seed', default=0, type=int,
                                 help='random seed (default: 0)')

    def parse_args(self):
        return self.parser.parse_args()


#######################################################################################################################
# self.add_arguments('train')
# self.add_arguments('val')
# self.add_arguments('test')

# def add_arguments(self, prefix):
#     self.parser.add_argument(f'--{prefix}_type', type=int,
#                              help=f'type of {prefix} dataset; if not specified, we will follow `dataset_type`')
#     self.parser.add_argument(f'--{prefix}_data_root', type=int,
#                              help=f'path of {prefix} dataset '
#                                   '(e.g. input/data_root/)')
#     self.parser.add_argument(f'--{prefix}_ann_file', type=str,
#                              help=f'path of annotation file for {prefix} dataset except for `train_data_root` '
#                                   f'(e.g. dataset/ImageSets/Main/{prefix}.txt)')
#     self.parser.add_argument(f'--{prefix}_img_prefix', type=str,
#                              help=f'prefix for all the {prefix} image samples '
#                                   '(e.g. dataset/)')
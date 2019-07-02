import argparse

class Opt:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Extract Video Feature')
        parser.add_argument('--root_dir', type=str, default=None, help='video directory.')
        parser.add_argument('--frame_unit', type=int, default=16, help='the number of frames in unit. Default: 16')
        parser.add_argument('--overlap', type=float, default=0.0, help='overlap percentage. Default: 0.0')
        parser.add_argument('--out_dir', type=str, default='./output/', help='output feature directory. Default: ./output/')
        parser.add_argument('--gpu_id', type=int, default=0, help='use GPU ID. Default: 0')
        parser.add_argument('--pretrained_path', type=str, default=None, help='load pretrained weight')
        parser.add_argument('--pretrain', type=str, default='sports-1m', choices=['sports-1m', 'kinetics'],
                            help='decide the number of class. If you specify sports-1m, then 487, else if kinetics, then 400. Default: sports-1m')
        parser.add_augment('--input_format', type=str, default='mp4', choice=['mp4', 'mov'],
                            help='specify input video format. you can choose mp4 or mov. Default: mp4.')
        parser.add_argument('--save_format', type=str, default='numpy', choice=['numpy', 'torch'],
                            help='specify save format. you can choose numpy or torch. If you choose numpy, save format is .npz, and if you choose torch, it is .pth.')
        parser.add_argument('--verbose', action='store_true', help='print saved file name')
        self.parser = parser

    def get_args(self):
        args = self.parser.parse_args()
        args.gpu_id = 'cuda:{}'.format(args.gpu_id)
        return self.parser.parse_args()

import argparse

class Opt:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Extract Video Feature')
        parser.add_argument('--root_dir', type=str, default=None, help='video directory.')
        parser.add_argument('--frame_unit', type=int, default=16, help='the number of frames in unit. Default: 16')
        parser.add_argument('--overlap', type=float, default=0.0, help='overlap percentage. Default: 0.0')
        parser.add_argument('--out_dir', type=str, default='./output/', help='output feature directory. Default: ./output/')
        parser.add_argument('--gpu_id', type=int, default=0, help='use GPU ID. Default: 0')
        parser.add_argument('--verbose', action='store_true', help='print save file')
        self.parser = parser

    def get_args(self):
        args = self.parser.parse_args()
        args.gpu_id = 'cuda:{}'.format(args.gpu_id)
        return self.parser.parse_args()

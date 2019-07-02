import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import glob
from fastprogress import progress_bar
import gc
import os
import sys
import json

import models
c3d = models.c3d

from loader import VideoLoader
from utils import convert_param
from options import Opt


def main():
    args = Opt().get_args()
    if args.root_dir is None:
        print('No define root_dir')
        sys.exit()

    if not torch.cuda.is_available():
        print('CUDA is not available')
        sys.exit()

    out_dir = args.out_dir
    visual_feature_out_dir = os.path.join(out_dir, 'visual-feature')
    activity_concept_out_dir = os.path.join(out_dir, 'visual-activity-concept')

    print('output directory : {}'.format(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(visual_feature_out_dir, exist_ok=True)
    os.makedirs(activity_concept_out_dir, exist_ok=True)

    # root_dir = glob.glob(os.path.join(args.root_dir,'*'))
    root_dir = glob.glob(os.path.join(args.root_dir,'*.'+args.input_format))

    if args.pretrain == 'sports-1m':
        model = c3d(pretrained=False)
        if args.pretrained_path is not None:
            model.load_state_dict(torch.load(args.pretrained_path))
    elif args.pretrain == 'kinetics':
        model = c3d(pretrained=False, num_class=400)
        if args.pretrained_path is not None:
            import torchfile
            model.load_state_dict(convert_param(torchfile.load(args.pretrained_path),
                                                model.state_dict(),
                                                verbose=True))

    model = model.to(args.gpu_id)
    model.eval()
    video_loader = VideoLoader(root_dir=root_dir, n_clip=args.frame_unit,
                               discard=True, to_tensor=True, overlap=args.overlap, resolution=(112, 112))
    with torch.no_grad():
        for name, unit in video_loader:
            unit = unit.unsqueeze(0)
            unit = unit.to(args.gpu_id)
            visual_feature, visual_activity_concept = model(unit)
            if args.save_format == 'numpy':
                # torch.cuda -> np.array
                visual_feature = visual_feature.cpu().detach().numpy()
                visual_activity_concept = visual_activity_concept.cpu().detach().numpy()
                # matrix to vector
                visual_feature = visual_feature.reshape(-1)
                visual_activity_concept = visual_activity_concept.reshape(-1)
                # need to pooling visual_activity_concept
                np.save(os.path.join(visual_feature_out_dir, name), visual_feature)
                np.save(os.path.join(activity_concept_out_dir, rename(name, args.video_format)), visual_activity_concept)
                if args.verbose:
                    print('save {}.npy'.format(name))
            elif args.save_format == 'torch':
                raise NotImplementedError


def rename(name, video_format):
    movie_name, st, e = name.split('_')
    st = st.split('.')[0]
    e = e.split('.')[0]
    new_name = movie_name + '.' +  video_format + '_' + str(int(st)) + '_' + str(int(e))
    return new_name

if __name__ == '__main__':
    main()
    print('success!')


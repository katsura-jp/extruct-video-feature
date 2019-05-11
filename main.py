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

import models.c3d as c3d
from loader import VideoLoader

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

    root_dir = glob.glob(os.path.join(args.root_dir,'*'))
    model = c3d(pretrained=True)
    model = model.to(args.gpu_id)
    model.eval()
    video_loader = VideoLoader(root_dir=root_dir, n_clip=args.frame_unit,
                               discard=True, to_tensor=True, overlap=args.overlap, resolution=(480, 640))
    with torch.no_grad:
        for name, unit in video_loader:
            unit = unit.unsqueeze(0)
            unit = unit.cuda(args.gpu_id)
            visual_feature, visual_activity_concept = model(unit)
            # torch.cuda -> np.array
            visual_feature = visual_feature.cpu().detach().numpy()
            visual_activity_concept = visual_activity_concept.cpu().detach().numpy()
            np.save(os.path.join('./tmp/visual-feature', name), visual_feature)
            np.save(os.path.join('./tmp/visual-activity-concept', name), visual_activity_concept)
            if args.verbose:
                print('save {}.npy'.format(name))

    print('success!')


if __name__ == '__main__':
    main()



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

import models
c3d = models.c3d

from loader import VideoLoader
from utils import convert_param,activity_name,pool_feature
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
    pooled_activity_concept_out_dir = os.path.join(out_dir, 'activity-concept')

    print('output directory : {}'.format(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(visual_feature_out_dir, exist_ok=True)
    os.makedirs(activity_concept_out_dir, exist_ok=True)
    os.makedirs(pooled_activity_concept_out_dir, exist_ok=True)

    root_dir = glob.glob(os.path.join(args.root_dir,'*'))

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
            # torch.cuda -> np.array
            visual_feature = visual_feature.cpu().detach().numpy()
            visual_activity_concept = visual_activity_concept.cpu().detach().numpy()
            # convert matrix to vector : (1,outdim) --> (outdim,)
            visual_feature = visual_feature.reshape(visual_feature.shape[1],)
            visual_activity_concept = visual_activity_concept.reshape(visual_activity_concept.shape[1],)
            # save feature
            np.save(os.path.join(visual_feature_out_dir, name), visual_feature)
            np.save(os.path.join(activity_concept_out_dir, activity_name(name)), visual_activity_concept)
            if args.verbose:
                print('save {}.npy'.format(name))

    # get movie name
    movie_names = []
    for movie in root_dir:
        movie_names.append(movie.split('/')[-1])

    for movie_name in progress_bar(movie_names):
        pool_feature(movie_name, activity_concept_out_dir, pooled_activity_concept_out_dir)


if __name__ == '__main__':
    main()
    print('success!')


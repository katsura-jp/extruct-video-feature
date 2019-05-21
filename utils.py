import torch
import numpy as np
import os

def convert_param(param7, param_py, verbose=False):
    params = param_py.copy()
    params_names = list(params.keys())
    idx = 0
    params7_names = ['cudnn.VolumetricConvolution', 'nn.Linear']

    for modules7 in param7['modules']:
        for module7 in modules7['modules']:
            module_name = module7.torch_typename().decode()
            if module_name in params7_names:
                # --- weight ---
                if verbose:
                    print(f'copy param : {params_names[idx]}')
                p = module7['weight']  # numpy.array
                p = torch.tensor(p, dtype=torch.float32)  # torch.tensor

                assert p.size() == params[params_names[idx]].size()

                params[params_names[idx]] = p
                idx += 1

                # --- bias ---
                if verbose:
                    print(f'copy param : {params_names[idx]}')
                p = module7['bias']  # numpy.array
                p = torch.tensor(p, dtype=torch.float32)  # torch.tensor

                assert p.size() == params[params_names[idx]].size()

                params[params_names[idx]] = p
                idx += 1

    return params


def activity_name(name):
    movie_name, start, end = name.split('_')
    new_name = movie_name + '.mp4_' + start.split('.')[0] + "_" + end.split('.')[0]
    return new_name

def pool(x, y):
    return (x + y) / 2.0


def get_file_name(movie_name, s, e):
    return f'{movie_name}_{s}_{e}.npy'

def pool_feature(movie_name, activity_dir, out_dir):
    s = 1
    step = 16
    nxt = 8
    while(1):
        path1 = activity_dir + get_file_name(movie_name, s, s+step) # 1-17
        path2 = activity_dir + get_file_name(movie_name, s+step, s+step+step) # 17-33
        if ((not os.path.exists(path1)) or (not os.path.exists(path2))):
            break
        else:
            f1 = np.load(path1)
            f2 = np.load(path2)
            pooled = pool(f1, f2)
            name = get_file_name(movie_name, s, s+step+step) # 1-33
            np.save(out_dir + name, pooled)
        s += nxt
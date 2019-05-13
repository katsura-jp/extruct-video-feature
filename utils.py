import torch


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
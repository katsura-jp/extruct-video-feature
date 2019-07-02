# Extract Video Feature
(I miss spell "extract" and this repository name is not correct spelling)
extruct video feature from C3D.

The output follows the [MAC](https://arxiv.org/abs/1811.08925).


### Requirements
- Python 3.7
- PyTorch 1.10
- torchfile 0.1.0
- others

### Model
- C3D

model is refered to [DavideA/c3d-pytorch](https://github.com/DavideA/c3d-pytorch).

and you can download pretrained weight(Sports-1M) of C3D from [c3d.pickle](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle)
and pretrained weight(Kinetics) of C3D from [c3d-sports1m-kinetics.t7](https://github.com/kenshohara/3D-ResNets/releases/download/1.0/c3d-sports1m-kinetics.t7).




### Option
- `--root_dir` (str) : give a directory path that have videos you want to extract feature.
- `--frame_unit` (int) : specify frame length input to model at once. Default: `16`.
- `--overlap` (float) : specify frame overlap percentage. If you specify 16 to frame_unit and 0.5 to overlap,
overlap frame is 8 frame(16*0.5), so start frame are 1,9, 17, 25,... Default : `0.0`.
- `--out_dir` (str) : specify the path to put output feature. Default: `./output`.
- `--gpu_id` (int) : specify GPU ID that you use. Default: `0`.
- `--pretrained_path` (str) : specify pretrained weight path.
- `--pretrain` (str) : specify pretrain dataset. you can choose `sport-1m` or `kinetics`. Default: `kinetics`.
- `--input_format` (str) : specify input video format. you can choose `mp4` or `mov`. Default: `mp4`.
- `--save_format` (str) : specify save format. you can choose `numpy` or `torch`. If you choose numpy, save format is `.npz`, and if you choose torch, it is `.pth`.
- `--verbose` (flag) : If you add this option, print saved file names.

### Usage
1. download weight.

2. 
```
python3 main.py --root_dir <video path> --frame_unit <unit> --overlap <Overlap> --gpu_id <GPU ID> --pretrained_path <path of c3d.pickle>
```
example
```
python3 main.py --root_dir /mnt/dataset/Charades --frame_unit 16 --overlap 0.5 --pretrained_path ./c3d.pickle --out_dir ./feature --verbose
```
### Save format
`.npy`

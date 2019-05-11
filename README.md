# Extruct Video Feature
extruct video feature from C3D.

The output follows the [MAC](https://arxiv.org/abs/1811.08925).


### Requirements
- Python 3.6
- PyTorch v1.0
- others

### Model
- C3D

model is refered to [DavideA/c3d-pytorch](https://github.com/DavideA/c3d-pytorch).

and you can download pretrained weight(Sports-1M) of C3D from [c3d.pickle](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle).

### Video Format
- mp4

### Option
- `--root_dir` (str) : give a directory path that have videos you want to extract feature.
- `--frame_unit` (int) : specify frame length input to model at once. Default: 16.
- `--overlap` (float) : specify frame overlap percentage. If you specify 16 to frame_unit and 0.5 to overlap,
overlap frame is 8 frame(16*0.5), so start frame are 1,9, 17, 25,... Default : 0.0.
- `--out_dir` (str) : specify the path to put output feature. Default: ./output
- `--gpu_id` (int) : specify GPU ID that you use. Default: 0.
- `--pretrained_path` (str) : specify pretrained weight path.
- `--verbose` (flag) : If you add this option, print saved file names.

### Usage
1. download weight.
- `python3.6 main.py --root_dir <video path> --frame_unit <unit> --overlap <Overlap> --gpu_id <GPU ID> --pretrained_path <path of c3d.pickle>`
(e.g. `python3.6 main.py --root_dir /mnt/dataset/Charades --frame_unit 16 --overlap 0.5 --pretrained_path ./c3d.pickle --out_dir ./feature --verbose`)

### Save format
nps

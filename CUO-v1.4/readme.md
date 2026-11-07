# Continual Low-Rank Adaptation via Cumulative Unified Optimization

## Environment

- GPU: NVIDIA GeForce RTX 4090
- Python: 3.12.4

```
torch==2.2.2
torchvision==0.17.2
timm==1.0.9
einops==0.8.0
ftfy==6.2.3
huggingface-hub==0.28.1
numpy==1.26.4
opencv-python==4.10.0.84
Pillow==10.4.0
regex==2024.7.24
scikit-image==0.25.2
scikit-learn==1.5.1
scipy==1.14.1
tqdm==4.67.1
```
These packages can be installed easily by
`pip install -r requirements.txt`

## Dataset preparation
### 1. Download the datasets and uncompress them:

- ImageNet-R: https://github.com/hendrycks/imagenet-r
- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- DomainNet: https://ai.bu.edu/M3SDA/
- ImageNet-A: https://github.com/hendrycks/natural-adv-examples

### 2. Rearrange the directory structure:

We use a unified directory structure for all datasets:
```
DATA_ROOT
    |- train
    |    |- class_folder_1
    |    |    |- image_file_1
    |    |    |- image_file_2
    |    |- class_folder_2
    |         |- image_file_2
    |         |- image_file_3
    |- val
         |- class_folder_1
         |    |- image_file_5
         |    |- image_file_6
         |- class_folder_2
              |- image_file_7
              |- image_file_8
```
We provide the scripts `split_[dataset].py` in the `tools` folder to rearange the directory structure.
Please change the `root_dir` in each script to the path of the uncompressed dataset.

## Training and evaluation
Using `bash run_scripts.sh` to run the following training scripts:

- 10-split ImageNet-R: `bash script_ir.sh   -t 10 --data_root "[PATH_TO_DATA_DIR]"`

- 20-split ImageNet-R: `bash script_ir.sh   -t 20 --data_root "[PATH_TO_DATA_DIR]"`

- 10-split CIFAR-100:  `bash script_c100.sh -t 10 --data_root "[PATH_TO_DATA_DIR]"`

- 10-split DomainNet:  `bash script_idn.sh  -t 10 --data_root "[PATH_TO_DATA_DIR]"`

- 20-split ImageNet-A: `bash script_ia.sh   -t 20 --data_root "[PATH_TO_DATA_DIR]"`

Please specify the `--data_root` argument in the above bash scripts to the locations of the datasets.
Change the `--seed` argument to use different seeds (e.g., 2026, 2027).

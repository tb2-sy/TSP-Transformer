# TSP-Transformer
## Abstract
Holistic scene understanding includes semantic segmentation, surface normal estimation, object boundary detection, depth estimation, etc. The key aspect of this problem
is to learn representation effectively, as each subtask builds
upon not only correlated but also distinct attributes. Inspired by visual-prompt tuning, we propose a Task-Specific
Prompts Transformer, dubbed TSP-Transformer, for holistic scene understanding. It features a vanilla transformer
in the early stage and tasks-specific prompts transformer
encoder in the lateral stage, where tasks-specific prompts
are augmented. By doing so, the transformer layer learns
the generic information from the shared parts and is endowed with task-specific capacity. First, the tasks-specific
prompts serve as induced priors for each task effectively.
Moreover, the task-specific prompts can be seen as switches
to favor task-specific representation learning for different
tasks. Extensive experiments on NYUD-v2 and PASCAL-Context show that our method achieves state-of-the-art performance, validating the effectiveness of our method for
holistic scene understanding.
### [ArXiv](https://arxiv.org/pdf/2311.03427.pdf) 

## Setup
Tested with PyTorch 1.11 and CUDA 11.3:
```bash
git clone https://github.com/tb2-sy/TSP-Transformer.git
conda create -n tsp python=3.7
conda activate tsp

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install tqdm Pillow easydict pyyaml imageio scikit-image tensorboard
pip install opencv-python==4.5.4.60 setuptools==59.5.0

pip install timm==0.5.4 einops==0.4.1
```

## Dataset
We use the same data (PASCAL-Context and NYUD-v2) as InvPT. You can download the data by:
```bash
wget https://data.vision.ee.ethz.ch/brdavid/atrc/NYUDv2.tar.gz
wget https://data.vision.ee.ethz.ch/brdavid/atrc/PASCALContext.tar.gz
```
And then extract the datasets by:
```bash
tar xfvz NYUDv2.tar.gz
tar xfvz PASCALContext.tar.gz
```
You need to specify the dataset directory as ```db_root``` variable in ```configs/mypath.py```. 

## Training
Set the config files in ```./conifigs```, with PASCAL-Context and NYUD-v2 dataset.
```
# Train the NYUD-v2 dataset
./run_nyud.sh

# Train the PASCAL-Context dataset
./run_pascal.sh
```
## Evaluation
```
# Evaluation the NYUD-v2 dataset
./infer_nyud.sh

# Evaluation the PASCAL-Context dataset
./infer_pascal.sh
```
## Pre-trained models
```
Please download the weights of our SoTA results.

```
|Version | Dataset | Download | Segmentation | Human parsing | Saliency | Normals | Boundary | 
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **TSP-Transformer (our paper)** | **PASCAL-Context** | [google drive]() | **81.48** | **70.64** | **84.86** | **13.69** | **74.80** | 
| TaskPrompter (ICLR 2023) | PASCAL-Context | - | 80.89 | 68.89 | 84.83 | 13.72 | 73.50 |
| InvPT (ECCV 2022) | PASCAL-Context | - | 79.03 | 67.61 | 84.81 | 14.15 | 73.00 |

|Version | Dataset | Download | Segmentation | Depth | Normals | Boundary|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **TSP-Transformer (our paper)** | **NYUD-v2** |[google drive]()| **55.39** | **0.4961** | **18.44** | 77.50 |
| TaskPrompter   (ICLR 2023) |NYUD-v2| - | 55.30 | 0.5152 | 18.47 | **78.20** | 
|InvPT (ECCV 2022) |NYUD-v2|-| 53.56 | 0.5183 | 19.04 | 78.10 |


## TODO/Future work
- [x] Upload paper and init project
- [x] Training and Inference code
- [x] Reproducible checkpoints
- [ ] Speed training and inference
- [ ] Reduce cuda memory usage

## Contact
For any questions related to our paper and implementation, please email wangshuo2022@shanghaitech.edu.cn.

## Citation
If you find our code or paper helps, please consider citing:
```
@article{wang2023tsp,
  title={TSP-Transformer: Task-Specific Prompts Boosted Transformer for Holistic Scene Understanding},
  author={Wang, Shuo and Li, Jing and Zhao, Zibo and Lian, Dongze and Huang, Binbin and Wang, Xiaomei and Li, Zhengxin and Gao, Shenghua},
  journal={arXiv preprint arXiv:2311.03427},
  year={2023}
}
```

## Acknowledgements
The code is available under the MIT license and draws from [InvPT](https://github.com/prismformore/Multi-Task-Transformer/tree/main/InvPT), [ATRC](https://github.com/brdav/atrc), and [MIT-Net](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch), which are also licensed under the MIT license.

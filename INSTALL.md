### Installation

We install different environments for single-task and all-in-one tasks.

For all-in-one tasks, please refer to [All-in-One](./All_in_One/INSTALL.md) directory

For single-task settings, 
1. Clone the repository
```
git clone https://github.com/c-yn/VIVNet.git
cd VIVNet
```

2. Create conda environment

```
conda create -n vivnet python=3.9
conda activate vivnet
```

3. Install dependencies
```
conda install pytorch=2.4.0 torchvision pytorch-cuda=12.4 -c pytorch
pip install opencv-python lmdb tqdm einops scipy scikit-image tensorboard natsort pyiqa joblib lpips scikit-learn
```

4. Install basicsr

```
python setup.py develop --no_cuda_ext
```


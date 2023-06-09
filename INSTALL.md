# Installation

## Requirements
- Python >= 3.6

- PyTorch >= 1.5

  We can successfully reproduce the main results under the setting below:

  Tesla **A100** (40G): CUDA 11.0 + PyTorch 1.7.1 + torchvision 0.8.2

- GCC >= 4.9

- cython: `pip install -U cython`

- simplejson: `pip install simplejson`

- PyAV: `conda install av -c conda-forge`

- iopath: `pip install -U iopath` or `conda install -c iopath iopath`

- psutil: `pip install psutil`

- OpenCV: `pip install opencv-python`

- pandas: `pip install pandas`

- tensorboard: `pip install tensorboard`

- moviepy: (optional, for visualizing video on tensorboard) `conda install -c conda-forge moviepy` or `pip install moviepy`

- timm: `pip install timm==0.4.12`


## Install Detectron2
```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```
If have to reinstall, first:
```
rm -rf build/ **/*.so
```

## Build

After having the above dependencies, run:
```
git clone https://github.com/MCG-NJU/EVAD.git
cd EVAD
python setup.py build develop
```

Now the installation is finished.

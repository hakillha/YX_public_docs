# mmdet_package

## Install mmlab

### System setup

Anaconda is highly recommended for all platforms.

Install the followings if you are unsure of their installations:

For CentOS
```shell
yum install git
yum install centos-release-scl
yum install devtoolset-8-gcc-c++
scl enable devtoolset-8 bash
```

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+
- PyTorch 1.1 or higher
- CUDA 9.0 or higher
- NCCL 2

  For CentOS
  ```shell
  yum localinstall $NCCL_RPM_FILE
  ```
- GCC 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)

  ```shell
  pip install pytest-runner
  pip install mmcv
  ```

### Install mmdetection

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
# remove "-c pytorch" if you are installing from the Tsinghua mirrors
```

c. Clone the mmdetection repository.

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```

d. Install build requirements and then install mmdetection.
(We install pycocotools via the github repo instead of pypi because the pypi version is old and not compatible with the latest numpy.)

```shell
pip install -r requirements/build.txt
pip install cython
pip install pycocotools
# or "pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI""
# and don't have to worry about the compatibility issue
pip install -v -e .  # or "python setup.py develop"
```

## Inference usage

Import `infer/cascade_infer.py`, `infer/det_infer.py` or `infer/seg_infer.py` depending on your task. Most part of these modules provide the same interfaces described in the following section.

#### 2020/02/21
For vehicle detection and fine-grained recognition we will only have to worry about module `det_infer.py` for now.

The vehicle body types we can recognize currently: [‘SUV’, ‘Sedan’, ‘Minivan’, ‘Microbus’, ‘Truck’, ‘Bus’].

### Interface description

det_infer.py:
```python
model_init(task_type=None, config_files=None, checkpoint_files=None)
'''
    args:
        config_files: A list of config file paths.
        checkpoint_files: A list of checkpoint file paths. Note you need to list them in order with the above list if you are using more than one model.
        task_type: You can safely ignore this.
    returns:
        A list of model instances.
'''

single_img_test(imgs, model=None, task_type='cars')
'''
    args:
        imgs: A list of test image paths. They need to be in JPEG format.
        model: A list of model instances which we will talk more about in detailed in the followings.
        task_type: Choose from ['p_generic', 'p_finegrained', 'cars'].
    returns:
        A list of dictionaries each of which is the output for one image, formatted as:
        [{'bbox': A list of bounding boxes with following format: [xmin, ymin, xmax, ymax],
        'score': A list of corresponding scores,
        'cls_name': A list of corresponding class names as strings},
        {...},
        {...},
        ...]
'''
```

We provide two types of usage for the above modules:

1. Hardcoding the model and config file paths in the inference script for your maximum eaze

    This way you will only have to worry about the ```single_img_test()``` interface and its ```imgs``` argument but you will have to change the code whenever you move the model/config files.

2. Or you can use ```model_init()``` to instantiate a model before looping through the images.

    That is, you build a list of model instances with ```model_init()``` first(so that you don't create them repeatedly) and feed it to ```single_img_test()``` in the loop. This way you don't need to change code in the inference module when you move the model files.

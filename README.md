# User related

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

    This way you will only have to worry about the `single_img_test()` interface and its `imgs``` argument but you will have to change the code whenever you move the model/config files.

2. Or you can use `model_init()``` to instantiate a model before looping through the images.

    That is, you build a list of model instances with `model_init()` first(so that you don't create them repeatedly) and feed it to `single_img_test()` in the loop. This way you don't need to change code in the inference module when you move the model files.

# Developer related

## Data preparation

### Execution environment

**\#\#\#TODO**

### Pipeline design breakdown

**\#\#\#TODO**

### Convert original data to COCO format

First, use `ftdet2coco.py` or `ftseg2coco.py` under `yx_toolset/FT_roadsign/python/` to convert all the original data into one single COCO format data file. This is done by setting the ratio of test set to 0 when using the above two scripts. Refer to the command line argument description in those scripts for more detailed info.

Example:
```shell
python ftdet2coco.py \
/media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/ \ 
--output_json_file /media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/yx_generated/coco_out.json
```

**Note**: The above two scripts were implemented to complete the whole data preparation pipeline but as of right now they are half deprecated and are only used for the aforementioned purpose. One of the effects from this change is that we no longer need to provide the class list metadata to this process. This process will transfer all the data uniformly to the COCO format file and leave the train/test set splitting, class filtering to next step.

### Set splitting and class processing

With the aggregated data file, use `ft_coco_postproc.py` under `yx_toolset/FT_roadsign/python/` to generate the final data file for training and testing. If you haven't done it, you will also have to create your own class processing logic in the `cls_filter()` interface and potentially also save the corresponding metadata in `post_proc_cfg_det.py` or `post_proc_cfg_seg.py`.

Example:
```shell
python ft_coco_postproc.py \
/media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/yx_generated/coco_out.json \
/media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/yx_generated/train.json \
/media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/yx_generated/test.json \
vehicle_logo_general \
--test_ratio 0.2
```

## Training under mmlab

**For a more detailed documentation please refer to [mmlab's own github page](https://github.com/open-mmlab/mmdetection) (especially their [GETTING_STARTED](https://github.com/open-mmlab/mmdetection/blob/master/docs/GETTING_STARTED.md) setction).**

Create a custom dataset file under `${MMDET_ROOT}\datasets`. If you have converted the data into COCO format like we did in the previous section, you can create a dataset class by simply inheriting the `CocoDataset` class of `mmdet` and only overwritting the classes. Here is an example:

```python
from .coco import CocoDataset
from .registry import DATASETS

@DATASETS.register_module
class FTDataset_det_exclude_p(CocoDataset):
    # Note the order of this set should follow the category id of your coco file
    CLASSES = ('i', 'panel', 'sc', 'tl', 'w') 
```

Import this class in `${MMDET_ROOT}\datasets\__init__.py` then you can use it in your config files and train on your own data.

Followings are the keyword-value that you generally would care about in the config files.

>**`dataset_type`** : The aforementioned custom dataset class;  
**`data_root`** : The root directory of the data.

In the `data` dictionary:
>**`ann_file`** in each set: The json file that stores the annotation data;  
**`img_prefix`** in each set: The directory where the images are. You don't need to seperate the images in different set since the COCO API can pick them out automatically;  
**`imgs_per_gpu`** : Change this to 1 if you run into graphical memory issue.

>**`pretrained`** : This points to a URL or a local directory to load an initial model from when your training begins. It's recommended to change this to point to a local file;  
**`num_classes`** in multiple heads: # of classes plus background class;  
**`work_dir`** : The directory where log files and checkpoint files are stored. **It's highly recommended to also store the config file in this directory. (Though you need to do this manually)** ;  
**`load_from`** : The directory that points to a checkpoint file when you are resuming a training. You should also be able to resume a training by passing this as an command line parameter;  
**`flip_ratio`** (flip augmentation related ones, this is relevant when mirrored classes are presented): You need to set this to `0` when mirrored classes are presented.

Additionally you may also care about soft-nms setting or multi-scale training setting (like `img_scale=[(1333, 640), (1333, 800)]` would enable a more diverse multi-scale training).

### Training

Train with multiple GPUs:
```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Example:
```shell
./tools/dist_train.sh \
/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x/cascade_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x.py \
1 --validate
```

# More info to be added in a more formal way
* Find advanced GCNET pretrained model files in [GCNET author github repo](https://github.com/xvjiarui/GCNet).
* The `mmdet` files in this repo is only for keeping track of changes on top of the original code since for some reason we can't effectively keep a copy of our own revised `mmdet` code on github. So please install `mmdet` using the official repo and add changes given the changes in this repo accordingly.
* If you run into `KeyError: 'segmentation'`, make following change to `${MMDET}/dataset/coco.py` :
  ```python
  if 'segmentation' in ann:
      gt_masks_ann.append(ann['segmentation'])
  ```

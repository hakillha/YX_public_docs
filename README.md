# YX pipeline

**IMPORTANT**: It's usually a good idea to look into the code for more detailed command line descryptions before you run any scripts in this repository even when some of them might have been explained in this documentation.

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

For consistency we have removed the hardcode stype usage.

### Vehicle attribute prediction

Install `timm` library for classification support:
```shell
cd ${mmdet_package}/pytorch-image-models
pip install -v -e .
```

We will cover how to use the vehicle attribute prediction APIs by going through what's going on when running `infer/cascade_infer.py` as a script. Following is the code snippet that you will care about:
```python
tasks = metadata.vehicle_attr_pred_cfg
for task in tasks:
    task.clsf_model = model_init(task.cfg, 'classify')
det_model = model_init(args.det_config_file, 'detection', args.det_ckpt_file)
imgs = glob(pj(args.img_folder_path, '*'))
for idx, img in enumerate(imgs):
    print('{}/{}'.format(idx, len(imgs)))
    res = single_img_test(img, det_model, tasks)
    print(res)
```
First you will need to change the `ckpt` fields in each of the `fg_task` class instance in `vehicle_attr_pred_cfg` in `infer\metadata.py`. Then you can build the classification models and store them in the `fg_task` instances like we did on line 1-3 in the above snippet. Line 4 builds the generic vehicle and logo detection model. After these you can pass the detection model and the task instances to the `single_img_test()` interface to run inference on the images.

The returned value of `single_img_test()` follows the format we define earlier (yes despite the name, this interface can run on a list of images and we haven't changed this FEATURE yet!):
```python
"""
A list of dictionaries each of which is the output for one image, formatted as:
            [{'bbox': A list of bounding boxes with following format: [xmin, ymin, xmax, ymax],
            'score': A list of corresponding scores,
            'cls_name': A list of corresponding class names as strings},
            {...},
            {...},
            ...]
"""
```
But since we add multi-task as a layer of additional logic, the above list will be embedded in yet another dictionary like the following example illustrates if we call the above result as `task_specific_res_${task}`:
```python
"""
{
    "color": task_specific_res_color,
    "car_make": task_specific_res_car_make,
    "body_type": task_specific_res_body_type
}
"""
``` 

An example of running this run as a script:
```shell
python cascade_infer.py --img_folder_path /home/yingges/data/vehicle/demo_imgs/ --det_config_file ~/data/model/0229_vehicle_attr/generic_veh_logo_det/cascade_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x.py --det_ckpt_file ~/data/model/0229_vehicle_attr/generic_veh_logo_det/epoch_6.pth 
```

### Segmentation

**###TODO**

# Developer related

## Data preparation

### Execution environment

**###TODO**

### Pipeline design breakdown

In short, the idea behind the data preparation pipeline I implement is that the two processes, the conversion from the data in the raw format to COCO format and the conversion from the COCO formatted data to data in a format that can directly be used to train, are separated to maximize the decoupling of the logics of these two processes.

**###TODO**

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

With the aggregated data file, use `ft_coco_postproc.py` under `yx_toolset/FT_roadsign/python/` to generate the final data file for training and testing. If you haven't done it, you will also have to create your own class processing logic in the `cls_filter()` interface and potentially also save the corresponding metadata in `post_proc_cfg_det.py` or `post_proc_cfg_seg.py`. Here is an example:
```python
# Changes in post_proc_cfg_det/seg.py file
```

```python
# Changes in ft_coco_postproc.cls_filter()
```

Processing example:
```shell
python ft_coco_postproc.py \
/media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/yx_generated/coco_out.json \
/media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/yx_generated/train.json \
/media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/yx_generated/test.json \
vehicle_logo_general \
--test_ratio 0.2
```
**###TODO: JSON file name extension check**

## Training under mmlab

**For a more detailed documentation please refer to [the github page of mmlab](https://github.com/open-mmlab/mmdetection) (especially their [GETTING_STARTED](https://github.com/open-mmlab/mmdetection/blob/master/docs/GETTING_STARTED.md) setction).**

Create a custom dataset file under `${MMDET_ROOT}\datasets`. If you have converted the data into COCO format like we did in the previous section, you can create a dataset class by simply inheriting the `CocoDataset` class of `mmdet` and only overwritting the classes. Here is an example:

```python
from .coco import CocoDataset
from .registry import DATASETS

@DATASETS.register_module
class FTDataset_det_exclude_p(CocoDataset):
    # Note the order of this set should be consistent with the category id of your COCO file
    CLASSES = ('i', 'panel', 'sc', 'tl', 'w') 
```

Import this class in `${MMDET_ROOT}\datasets\__init__.py` then you can use it in your config files and train on your own data.

Followings are the keyword-value that you generally would care about in the config files:

>**`dataset_type`** : The aforementioned custom dataset class;  
**`data_root`** : The root directory of the data.

In the `data` dictionary:
>**`ann_file`** in each set: The json file that stores the annotation data;  
**`img_prefix`** in each set: The directory where the images are. You don't need to seperate the images in different set since the COCO API can pick them out automatically;  
**`imgs_per_gpu`** : Change this to 1 if you run into graphical memory issue.

And:
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

### Testing

Analyze logs and find the best checkpoint:
```shell
python tools/analyze_logs.py plot_curve \
/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x/20200224_170512.log.json  \
--keys bbox_mAP_50
```

Test:
```shell
python tools/test.py \
/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x/cascade_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x.py \
/media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x/epoch_6.pth \
--json_out /media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x/epoch_6
```

## Training a classification model for finegrained task

Refer to [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models) for the installation instructions of the classification model.

In our case we need to crop out results of a detector to provide data for the classification model (if you already have the data in classification format, ignore this step):
```shell
python classification_utils.py patch_extract \
--input_path /media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/ \ 
--output_path /media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/clsf/
```

The dataset should follow the file structure below: 
```
${ROOT}
    train\
        ${class1}\
        ${class2}\
        ${class3}\
        ...
    val\
        ${class1}\
        ...
```

**###TODO: Command line parameters sanity check**

Train/test set splitting example:
```shell
python classification_utils.py set_split \
--input_path /media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/clsf/body_type_train/ \
--output_path /media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/clsf/body_type_test/ \
--test_ratio .2
```

### Environment setup

### Training

Command line parameters you should care about:
>**`batch_size`**  
**`num-class`**  
**`pretrained`** : Use a pretrained model to start the training;  
**`output`** : Output directory. Note that the code will automatically create a unique subfolder so you only need to point this to a general work directory. 

Depending on your case there might be a couple settings you would care about. 

**Update:** The following doesn't seem to be true.  
Input size is one of the examples: you shouldn't have an input size too much larger than the data image size.

```shell
python train.py /media/yingges/Data_
Junior/data/vehicle/FT/cljgh_cut_20200220_checked/clsf/body_type/ --model efficientnet_b0 -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.4 --drop-connect 0.3 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .016 --num-class 15 --pretrained --output /media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/clsf/work_dir/
```
When prompted with "Downloading...", it's recommended to use a VPN and a web browser to download the linked pretrained model to the target location.

## Evaluation and visualizatoin

Evaluation example:
```shell
python eval_main.py eval \
--anno_file_path /media/yingges/Data_Junior/data/vehicle/FT/cljgh_cut_20200220_checked/yx_generated/test.json \
--res_file_path /media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x/epoch_6.bbox.json
```

Visualization example:
```shell
python eval_main.py viz --ann_type segm \
--anno_file_path /media/yingges/Data/201910/FT/FTData/yunxikeji-01-2019-10-21/Normal/test.json \
--res_file_path /media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_mask_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_0225/epoch_11_eval/epoch_11.segm.json \
--img_folder_path /media/yingges/Data/201910/FT/FTData/yunxikeji-01-2019-10-21/images/ \
--output_save_path /media/yingges/Data_Junior/test/12/mmdet_workdirs/cascade_mask_rcnn_r4_gcb_dconv_c3-c5_x101_32x4d_fpn_syncbn_1x_0225/epoch_11_eval/viz/
```

# Miscellaneous utilities

### Subset extractaction
**###TODO**

# Informal additional info and notes
* Find advanced GCNET pretrained model files in [GCNET author github repo](https://github.com/xvjiarui/GCNet).
* The `mmdet` files in this repo is only for keeping track of changes on top of the original code since for some reason we can't effectively keep a copy of our own revised `mmdet` code on github(compilation doesn't work after pulling it into a new environment). So please install `mmdet` using the official repo and make changes based on the changes in this repo. Also if you are just using this repo for inference then you don't need to worry about this and can simply install mmlab from the official repo.
* If you run into `KeyError: 'segmentation'` after starting a detector training, make the following change to `${MMDET}/dataset/coco.py` :
  ```python
  if 'segmentation' in ann:
      gt_masks_ann.append(ann['segmentation'])
  ```
* Since we enlarge the results of a detector before feeding them into a classifier. We can use a similar pre-process when combining detection and classification as a mutli-task problem.
* ```shell
  python extract_subset.py extract_subset \
  --source_path /media/yingges/Data/201910/FT/FTData/yunxikeji-01-2019-10-21/ \
  --output_path=/media/yingges/Data/201910/FT/FTData/yunxikeji-01-2019-10-21/sample/backup_team/
  ```

# A generalizable 3D framework and model for self-supervised learning in medical imaging

***npj Digital Medicine (2025)***

<img src="assets/3dino_logo.png" alt="3DINO logo" width="200px"/>

Tony Xu, 
Sepehr Hosseini, 
Chris Anderson, 
Anthony Rinaldi, 
Rahul G. Krishnan, 
Anne Martel,
Maged Goubran

[Paper](https://doi.org/10.1038/s41746-025-02035-w) | [Preprint](https://arxiv.org/abs/2501.11755) | [HF Weights](https://huggingface.co/AICONSlab/3DINO-ViT)

This codebase contains PyTorch code and our 3DINO-ViT pretrained model for the 3DINO self-supervised framework for training networks on unlabled 3D medical images, developed at Sunnybrook Research Institute.   

**Abstract:** Current self-supervised learning methods for 3D medical imaging rely on simple pretext formulations and organ- or modality-specific datasets, limiting their generalizability and scalability. We present 3DINO, a cutting-edge SSL method adapted to 3D datasets, and use it to pretrain 3DINO-ViT: a general-purpose medical imaging model, on an exceptionally large, multimodal, and multi-organ dataset of ~100,000 3D medical imaging scans from over 10 organs. We validate 3DINO-ViT using extensive experiments on numerous medical imaging segmentation and classification tasks. Our results demonstrate that 3DINO-ViT generalizes across modalities and organs, including out-of-distribution tasks and datasets, outperforming state-of-the-art methods on the majority of evaluation metrics and labeled dataset sizes. Our 3DINO framework and 3DINO-ViT will be made available to enable research on 3D foundation models or further finetuning for a wide range of medical imaging applications.  

![Gif of 3DINO PCA visualization on brain MRI](assets/pca_example.gif)

## Installation

3DINO code runs on Python 3.9. Clone the codebase, then use the provided `requirements.txt` file and [pip](https://pip.pypa.io/en/stable/getting-started/) to install the necessary libraries for this repo:

```shell
pip install -r requirements.txt
```

## 3DINO-ViT Model

The 3DINO-ViT pretrained weights are now available for download on [HuggingFace](https://huggingface.co/AICONSlab/3DINO-ViT)! 

A barebones example for loading the pretrained network and applying it to an image to extract a feature vector representation is provided in [this notebook](notebooks/basic_model_use.ipynb).

## Pretraining

We provide code to pretrain 3DINO on general 3D medical imaging datasets. We use [MONAI](https://monai.io/) for data loading, so any format that can be loaded by the [LoadImage](https://docs.monai.io/en/stable/transforms.html#loadimage) transform can be used.
Datasets that we pretrained on can be found in the paper.

### Pretraining Dataset

Datasets should be formatted as a list of dictionaries with the following format. 
The `image` key should point to the path of the image file, the `shape` key should contain the shape of the image (e.g. when calling `loaded_img.shape`), and the `spacing` key should contain the voxel spacing of the image in arbitrary units (but consistent per image).
The `shape` and `spacing` keys are needed for 3D random resized cropping.  

```python
[
    {
        "image": "path/to/image1.nii.gz",
        "shape": [128, 128, 64],
        "spacing": [0.5, 0.5, 1.0],
    },
    {
        "image": "path/to/image2.nii.gz",
        "shape": [256, 256, 128],
        "spacing": [0.7, 0.7, 1.0],
    },
    ...
]
```

Save this as a JSON file, and adjust `dataset_path` in the config file to point to this JSON file.

### Standard Pretraining

Standard pretraining can be run using the following command for a single node with 4 A100-80GB GPUs:

```shell
PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node 4 --master_port 29501 dinov2/train/train3d.py \
  --config-file 'dinov2/configs/ssl3d_default_config.yaml' \
  --output-dir 'path/to/output_dir' \
  --cache-dir 'path/to/cache_dir' 
```

The `cache-dir` argument is used for MONAI [CacheNTransDataset](https://docs.monai.io/en/stable/data.html#cachentransdataset) caching.
This dataset saves images after a few preprocessing transforms to disk (potentially in a faster temporary storage system if training on a SLURM cluster).
We found this to greatly speed up loading. Remove this argument if you do not want to use caching.
Pretraining for 125000 iterations with a ViT-Large took approximately 10 days on 4 A100-80GB GPUs.

The training code saves the weights of the teacher in the `eval` folder every 12500 iterations.

### High Resolution Adaptation

To perform high resolution adaptation on the pretrained network, use the following command. 
Adjust `full_pretrained_weights` in the config file to point to the saved teacher weights from standard pretraining.

```shell
PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node 4 --master_port 29501 dinov2/train/train3d.py \
  --config-file 'dinov2/configs/train/vit3d_highres.yaml' \
  --output-dir 'path/to/highres_output_dir' \
  --cache-dir 'path/to/cache_dir' 
```

High-res adaption for 12500 iterations with a ViT-Large took approximately 1 day on 4 A100-80GB GPUs.

## Finetuning

The training code regularly saves the teacher weights. In order to evaluate the model, run the following evaluation on a single node:

### Finetuning dataset setup

Datasets should be formatted as a dict with `training`, `validation`, and `test` keys each with a list of dictionaries, as in the following format:
    
```python
{
    "training": [
        {
            "image": "path/to/train/image1.nii.gz",
            "label": "path/to/train/label1.nii.gz",  # or int for classification
        },
        ...
    ],
    "validation": [
        {
            "image": "path/to/val/image1.nii.gz",
            "label": "path/to/val/label1.nii.gz",
        },
        ...
    ],
    "test": [
        {
            "image": "path/to/test/image1.nii.gz",
            "label": "path/to/test/label1.nii.gz",
        },
        ...
    ]
}
```

Save this as a JSON file with name `<dataset_name>_100_datalist.json`, and adjust the `base-data-dir` argument below to point to the directory where it is saved.

### Segmentation finetuning

To finetune the model for segmentation, use the following command.
There are `UNETR`, `ViTAdapterUNETR`, and `Linear` segmentation heads available.

`ViTAdapterUNETR` uses our 3D adaptation of the original [ViT-Adapter](https://github.com/czczup/ViT-Adapter) module, which injects spatial information into pretrained ViTs. 
This module was also used in the original DINOv2 when performing segmentation tasks.  

```shell
PYTHONPATH=. python dinov2/eval/segmentation3d.py \
  --config-file 'dinov2/configs/train/vit3d_highres.yaml' \
  --output-dir  'path/to/output_dir' \
  --pretrained-weights 'path/to/eval/training_12499/teacher_checkpoint.pth' \
  --dataset-name 'BraTS' \
  --dataset-percent 100 \
  --base-data-dir 'path/to/finetuning/jsonfile/base_dir' \
  --segmentation-head 'ViTAdapterUNETR' \
  --epochs 100 \
  --epoch-length 300 \
  --eval-iters 600 \
  --warmup-iters 3000 \
  --image-size 112 \
  --batch-size 2 \
  --num-workers 20 \
  --learning-rate 1e-4 \
  --cache-dir 'path/to/cache_dir' \
  --resize-scale 1.0
```

### Adding custom segmentation datasets

Adding new segmentation datasets for finetuning requires the following steps:

1. Create a new dataset name and save the json file as described.   
2. Create training and validation transforms for the dataset in: `dinov2/eval/segmentation_3d/augmentations.py`
3. Create new evaluation metrics for the dataset in: `dinov2/eval/segmentation_3d/metrics.py`
4. Create a loss function to train the network in: `dinov2/eval/segmentation3d.py` (line 214)
5. Create a new segmentation dataset in: `dinov2/data/loaders.py`

### Classification finetuning

To finetune the model for classification, use the following command.

```shell
PYTHONPATH=. python dinov2/eval/linear3d.py \
  --config-file 'dinov2/configs/train/vit3d_highres.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/eval/training_12499/teacher_checkpoint.pth' \
  --dataset-name 'COVID-CT-MD' \
  --dataset-percent 100 \
  --base-data-dir 'path/to/finetuning/jsonfile/base_dir' \
  --epochs 100 \
  --epoch-length 125 \
  --save-checkpoint-frequency 50 \
  --eval-period-iterations 50 \
  --image-size 112 \
  --batch-size 32 \
  --num-workers 10 \
  --dataset-seed 0 \
  --cache-dir 'path/to/cache_dir'
```

### Adding new classification datasets

Adding new classification datasets for finetuning requires the following steps:
1. Create a new dataset name and save the json file as described.
2. Create training and validation transforms for the dataset in: `dinov2/data/transforms.py`
3. Create a new classification dataset in: `dinov2/data/loaders.py`

## Unsupervised Visualization

We provide example codes to generate unsupervised visualizations on an input image.
First, run the following command using the pretrained model to generate 3D representations of the image (`vis-type` can be `mhsa` or `pca`):
Then, use the provided [notebook](notebooks/unsupervised_vis.ipynb) to visualize the outputs. 

```shell
PYTHONPATH=. python dinov2/eval/vis_pca.py \
  --config-file 'dinov2/configs/train/vit3d_highres.yaml' \
  --output-dir 'path/to/output_vis_dir' \
  --pretrained-weights 'path/to/eval/training_12499/teacher_checkpoint.pth' \
  --image-path 'path/to/image.nii.gz' \
  --vis-type 'mhsa' \
  --input-type 'full_image'
```

## License
3DINO code and 3DINO-ViT weights are released under the CC BY-NC-ND 4.0 license.

✅ **You MAY:**

- Use this framework for **academic, research, and educational purposes**.
- Share or redistribute the original, **unmodified** version of this framework with proper attribution as detailed below.

❌ **You MAY NOT:**

- Use this framework for **commercial purposes** (as defined below).
- Modify, adapt, or create derivative works based on this framework.
- Distribute a modified version of this framework.

For full license details, refer to the official **[CC BY-NC-ND 4.0 License](https://creativecommons.org/licenses/by-nc-nd/4.0/)**

By Commercial Purposes, we mean that this framework **may not** be used:

- By **for-profit entities** for internal research, product development, or services.
- In **industry-funded** or **corporate-sponsored** research.
- As part of **commercially funded academic projects** without prior approval.
- In any project where the results will be used for **monetary gain** (e.g., patent filings, proprietary software development, licensing to industry).

If you are unsure whether your use qualifies as non-commercial, contact **maged.goubran@utoronto.ca**.

## Disclaimer

This software is provided **"as is"** without warranty of any kind. Sunnybrook Research Institute makes no representations or guarantees regarding its accuracy, reliability, performance, or suitability for any particular purpose. Users assume full responsibility for its use and application.

## Contact

For inquiries regarding permissions, exceptions, or licensing, contact **maged.goubran@utoronto.ca**.

## Acknowledgements

This repo builds upon the excellent work from the original [DINOv2](https://github.com/facebookresearch/dinov2) and [ViT-Adapter](https://github.com/czczup/ViT-Adapter) for 2D natural images.

## Citing 3DINO

If you find this repository useful or use 3DINO-ViT in your research, please consider giving a star and citing the following paper:

```
@article{xu3dino2025,
  title={A generalizable 3D framework and model for self-supervised learning in medical imaging},
  author={Xu, Tony and Hosseini, Sepehr and Anderson, Chris and Rinaldi, Anthony and Krishnan, Rahul G. and Martel, Anne L. and Goubran, Maged},
  journal={npj Digital Medicine},
  year={2025},
  doi={10.1038/s41746-025-02035-w},
}
```
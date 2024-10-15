# DICOM 2D - YOLOv5 Object Detection

This repository contains a pipeline for object detection with YOLOv5 and DICOM 2D medical images using PyTorch. This branch contains a modified version of the original [repository from Ultralytics](https://github.com/ultralytics/yolov5). 

## Table of Contents
- [Introduction](#introduction)
- [DICOM 2D Images](#dicom-2d-images)
- [Object Detection](#binary-image-classification)
- [How to Use](#how-to-use)
  - [Dataset Preparation](#dataset-preparation)
  - [Configuration File](#configuration-file)
  - [Training, Validation, and Exporting the Model](#training-validation-and-exporting-the-model)

## Introduction

This repository is based on the original [YOLOv5]((https://github.com/ultralytics/yolov5)) object detection framework, extended to support DICOM 2D medical images. The dev and read_change branch introduces functionality to process and detect objects within DICOM images, making it easier to apply state-of-the-art computer vision techniques in the medical field. It takes place specifically on the dev and read_change branches.

## DICOM 2D Images

DICOM (Digital Imaging and Communications in Medicine) is a standard format for storing medical imaging data. In this project, we work with DICOM files, which store 2D medical images such as X-rays, MRIs, or CT scans. DICOM images are widely used in medical environments because they contain both the image data and metadata (e.g., patient information, imaging parameters).

To handle DICOM images, we convert them into 2D image tensors that can be fed into the YOLOv5 model for object detection tasks.

## Object Detection

Object detection is a computer vision technique that involves identifying and locating objects within an image or video. Unlike image classification, which only labels the entire image, object detection goes a step further by predicting both the classes of objects present and their precise locations, usually in the form of bounding boxes.

Key Components of Object Detection:
- Classification: Identifying the type or class of each object (e.g., cat, car, person).
- Localization: Determining the exact position of each object within the image.

## How to Use

### Dataset Preparation

Before starting the training process, you need to prepare and organize your dataset. The images should be arranged in the following structure:

```plaintext
/dataset
│
├── /images
│   ├── /train
│   │   └── image1.dcm, image2.dcm, ...
│   ├── /val
│       └── image1.dcm, image2.dcm, ...
│
├── /labels
│   ├── /train
│   │   └── image1.txt, image2.txt, ...
│   ├── /val
│       └── image1.txt, image2.txt, ...
│
└── config_data.yaml
```

- `images/`: This folder contains the training and validation images organized into subfolders.
- `labels/`: This folder contains the training and validation labels or annotations for each image.

### Configuration File

The configuration file should looks like:

Example of a configuration file (`test.yaml`):

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/dicom  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/val  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
names:
  0: pneumonia
```

### Training, Validation, and Exporting the Model

After setting up the dataset and configuration file, you can call the Python scripts to start the training, validation, and model exporting process. The instruction of use are located in the Jupyter Notebook attached in the repository for ease of use, experimentation, and quick model iterations.

To use the notebook:

1. Open the provided notebook located in Notebooks folder (Sandbox_DICOM2D_Yolov5_modified_version.ipynb) in your Jupyter environment or Colab.
2. Update the paths for your dataset and configuration file.
3. Run through the cells to:
   - Setting up.
   - Train the model.
   - Validate the model.
   - Export the trained model to a file format such as `.torchscript` or `.onnx`.

The notebook is structured to walk you through each step of the process interactively. Additionally, from the original repository there is another Notebook which is localted in the root of the repository and the name is tutorial.ipynb.


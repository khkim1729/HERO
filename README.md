# HERO: Hybrid PET-CT Representation Learning for Head and Neck Tumor Segmentation and Outcome Prediction

## Overview

**HERO** is a deep learning framework for multimodal medical image analysis using PET-CT data from the HECKTOR challenge dataset.  
The goal of this project is to perform head and neck tumor segmentation and outcome prediction by learning complementary anatomical and metabolic representations from CT and PET images.

This repository is prepared for research on:

- Head and neck tumor segmentation
- PET-CT multimodal representation learning
- Outcome prediction using imaging-based features
- HECKTOR challenge dataset analysis

## Project Title

**HERO: Hybrid PET-CT Representation Learning for Head and Neck Tumor Segmentation and Outcome Prediction**

## Dataset

This project uses the **HECKTOR dataset**, which contains PET-CT images of patients with head and neck cancer.

The dataset includes:

- CT images
- PET images
- Tumor segmentation masks
- Clinical/outcome-related information, depending on the challenge year

Due to data usage restrictions, the dataset is not included in this repository.  
Please download the dataset from the official HECKTOR challenge page.

## Repository Structure

```bash
HERO/
├── data/
│   └── README.md
├── configs/
├── datasets/
├── models/
├── preprocessing/
├── train.py
├── test.py
├── inference.py
├── requirements.txt
└── README.md

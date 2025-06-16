# YOLOv8 VisDrone Object Detection and Tracking

This project implements object detection and tracking for drone footage using a fine-tuned YOLOv8 model on the VisDrone dataset, combined with DeepSORT for multi-object tracking.

## Overview

The project consists of two main components:
1. **Object Detection**: YOLOv8 model fine-tuned on the VisDrone2018 dataset
2. **Object Tracking**: DeepSORT algorithm for multi-object tracking in video sequences

## Dataset Information

The model was trained on the VisDrone2018 dataset for object detection in images. The dataset contains three distinct sets with no overlap:

| Dataset | Training | Validation | Test-Challenge |
|---------|----------|------------|----------------|
| Object detection in images | 6,471 images | 548 images | 1,580 images |

**Dataset Source**: [VisDrone2018-DET-toolkit](https://github.com/VisDrone/VisDrone2018-DET-toolkit)

## Supported Classes

The model is trained to detect the following 10 object classes:
- pedestrian
- people
- bicycle
- car
- van
- truck
- tricycle
- awning-tricycle
- bus
- motor

### Tracking Results
The tracking performance achieved using DeepSORT and on this dataset [VisDrone2018-MOT-toolkit repository](https://github.com/VisDrone/VisDrone2018-MOT-toolkit).:

| Metric | Score |
|--------|-------|
| MOTA (Multiple Object Tracking Accuracy) | 27.11% |
| MOTP (Multiple Object Tracking Precision) | 16.44% |
| IDF1 (ID F1 Score) | 49.69% |
| Precision | 69.91% |
| Recall | 61.93% |

## Installation and Setup

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone (https://github.com/abdelrahmanHamdyG/Visdrone-yolov8-tracking)
   cd Visdrone-yolov8-tracking
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the fine-tuned model**
   
   Download the fine tuned model from Google Drive:
   [Download Model](https://drive.google.com/file/d/1Hrp8R2eaHqUIQT5yMNM8SFpY--xCqhYT/view?usp=sharing)
   
   Place the downloaded model file in the project root directory.

## Usage

### Running Object Tracking

1. **Configure the sequence directory**
   
   Edit the `track.py` file and modify the `sequence_dir` parameter to point to your video sequence:
   ```python
   sequence_dir="VisDrone2019-MOT-test-dev/sequences/uav0000249_00001_v"
   ```

2. **Run the tracking script**
   ```bash
   python track.py
   ```

## Technical Details

### YOLOv8 Fine-tuning
- Base model: YOLOv8
- Training dataset: VisDrone2018 detection dataset
- Fine-tuning approach: Transfer learning from COCO pre-trained weights
- Target classes: 10 object classes relevant to drone surveillance

### DeepSORT Tracking
- Object association: Deep learning-based appearance features
- Kalman filtering: For motion prediction and tracking
- Hungarian algorithm: For optimal assignment of detections to tracks

## Evaluation

The model performance is evaluated using standard MOT (Multiple Object Tracking) metrics:
- **MOTA**: Measures tracking accuracy considering false positives, false negatives, and identity switches
- **MOTP**: Measures tracking precision based on bounding box overlap
- **IDF1**: Harmonic mean of identification precision and recall
- **Precision/Recall**: Standard detection metrics

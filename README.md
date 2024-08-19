# Overview

This project presents a deep learning framework for predicting seizures in patients with Temporal Lobe Epilepsy (TLE). The research utilizes advanced memory-based learning approaches, including transfer learning and Recurrent Neural Networks (RNNs), combined with continual learning techniques to enhance model adaptability. The study employs EEG signals and a comprehensive set of deep learning models to classify seizures with high accuracy.

# Problem Statement

Epilepsy is a neurological disorder characterized by unpredictable seizures, affecting millions globally. Traditional methods of predicting seizures using EEG (Electroencephalography) data often fall short in accuracy and real-time applicability. This project aims to improve seizure prediction using deep learning algorithms, focusing on Temporal Lobe Epilepsy, which is one of the most common types of epilepsy.

# Methodology

The framework integrates various phases, including data acquisition, preprocessing (signal denoising), model training using memory-based algorithms (transfer learning and RNNs), and continual learning to avoid catastrophic forgetting. The EEG dataset was preprocessed using hybrid filters, with the Butterworth Wavelet denoising filter achieving the highest Signal-to-Noise Ratio (SNR). Googlenet was identified as the best-performing model during transfer learning, achieving 97.5% accuracy which was further enhanced to 98.7% using Elastic Weight Consolidation (EWC) for continual learning.

# Data Pre-processing

**Instances:** 11,500

**Features:** 179 columns

**Classes:** Five classes, including epileptic and non-epileptic conditions.

The preprocessing involved:

Filling missing values.

Filtering noise using hybrid filters.

Selecting the Butterworth Wavelet denoising filter based on its SNR value of 24.11 dB.

# Model Training

Transfer Learning Models: Googlenet, Resnet, VGG, Densenet, and Alexnet.
Recurrent Neural Networks (RNNs).
Googlenet outperformed other models with an accuracy of 97.5%, making it the primary model for further refinement through continual learning.

**Continual Learning**

Elastic Weight Consolidation (EWC) was implemented to enhance model robustness and adaptability to new data. The custom learning rate scheduler dynamically adjusted the learning rate, resulting in improved training performance and overall accuracy.

# Implementation Workflow

Data Acquisition: Retrieve EEG dataset.

Preprocessing: Clean and denoise data using hybrid filters.

Model Training: Train models using transfer learning and RNNs.

Continual Learning: Apply EWC for improved accuracy and model stability.

# Results

Best Filter: Butterworth Wavelet denoising filter with SNR of 24.11 dB.

| S.No. | Model      | Accuracy (in %) | F-1 score |
|-------|------------|-----------------|-----------|
| 1.    | Alexnet    | 94.1            | 0.94      |
| 2.    | Densenet   | 96.7            | 0.91      |
| 3.    | Googlenet  | 97.5            | 0.97      |
| 4.    | VGG        | 97.0            | 0.97      |
| 5.    | Resnet     | 97.5            | 0.97      |
| 6.    | RNN        | 87.8            | 0.70      |

Best Model: Googlenet achieving 98.7% accuracy after EWC continual learning technique.

# Future Scope

The framework can be extended by integrating multimodal data, incorporating Explainable AI (XAI) for better interpretability, and leveraging Generative AI for data augmentation. Future work could also explore real-time deployment for clinical applications.

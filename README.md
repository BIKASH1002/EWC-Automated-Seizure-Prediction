# Epileptic Seizure Prediction Using Deep Learning and Continual Learning

<div  align = "justify">
    
# Overview

This project presents a deep learning framework for predicting seizures in patients with Temporal Lobe Epilepsy (TLE). The project utilizes advanced memory-based learning approaches, including transfer learning and Recurrent Neural Networks (RNNs), combined with continual learning techniques to enhance model adaptability. The work employs EEG signals and a comprehensive set of deep learning models to classify seizures with high accuracy.

# Problem Statement

Epilepsy is a neurological disorder characterized by unpredictable seizures, affecting millions globally. Traditional methods of predicting seizures using EEG (Electroencephalography) data often fall short in accuracy and real-time applicability. This project aims to improve seizure prediction using deep learning algorithms, focusing on Temporal Lobe Epilepsy, which is one of the most common types of epilepsy.

# Data Understanding

The dataset contains EEG signals recorded over time from patients diagnosed with epilepsy. It includes approximately 11,500 instances and 179 columns, capturing both epileptic and non-epileptic episodes. The dataset is publicly available and can be accessed from repositories like UCI or Kaggle.

The dataset is time-series based, which is critical for capturing temporal patterns in EEG signals. The presence of multiple features, including electrical signal amplitude, allows the models to learn complex patterns essential for seizure prediction. Additionally, the dataset is well-suited for applying transfer learning models and continual learning strategies.

# Methodology

The framework integrates various phases, including data acquisition, preprocessing (signal denoising), model training using memory-based algorithms (transfer learning and RNNs), and continual learning to avoid catastrophic forgetting. The EEG dataset was preprocessed using hybrid filters, with the Butterworth Wavelet denoising filter achieving the highest Signal-to-Noise Ratio (SNR). Googlenet was identified as the best-performing model during transfer learning, achieving 97.5% accuracy which was further enhanced to 98.7% using Elastic Weight Consolidation (EWC) for continual learning.

# Setup 

**Jupyter Notebook:** for development

**Libraries:** numPy, pandas, matplotlib, scikit-learn, pywt, tensorflow

# Models Used

**1)** Transfer Learning Models: Googlenet, Resnet, VGG, Densenet and Alexnet

**2)** Recurrent Neural Networks (RNNs)

**3)** Continual Learnining: Elastic Weight Consolidation (EWC)

<div  align = "justify">

# Design

<p align="center">
    <img src="https://github.com/user-attachments/assets/6407f63c-b797-4eb5-b1fc-7db885266c2c" alt="Design"/>
</p>

# Implementation

<div  align = "justify">
    
**1) Data Preprocessing:**
Handle missing values through mean imputation. Noise filtering using hybrid filters, with the Butterworth-Wavelet denoising filter achieving the highest signal-to-noise ratio (24.11 dB).

**2) Model Training:**
Apply memory-based models like RNN and transfer learning models (GoogLeNet, ResNet). Evaluate performance based on accuracy, F1-score, and other metrics.
GoogLeNet achieved the best results in initial training with 97.5% accuracy.

**3) Continual Learning:**
Implement Elastic Weight Consolidation (EWC) to improve the model’s adaptability while retaining previous knowledge. Fine-tune the model with a custom learning rate scheduler to enhance overall accuracy to 98.7%.

</div>

# Results

Best Filter: Butterworth Wavelet denoising filter with SNR of 24.11 dB.

<p align="center">
    <img src="https://github.com/user-attachments/assets/2f010fa5-25ed-4f66-af19-aeeb196d7d8d" alt="Filter analysis and comparison"/>
</p>

<div align = "center">
  
| S.No. | Model      | Accuracy (in %) | F-1 score |
|-------|------------|-----------------|-----------|
| 1.    | Alexnet    | 94.1            | 0.94      |
| 2.    | Densenet   | 96.7            | 0.91      |
| 3.    | Googlenet  | 97.5            | 0.97      |
| 4.    | VGG        | 97.0            | 0.97      |
| 5.    | Resnet     | 97.5            | 0.97      |
| 6.    | RNN        | 87.8            | 0.70      |

</div>

**GRAPHICAL ANALYISIS**

<div  align = "center">
<table>
    <tr>
        <td>
            <img src="https://github.com/user-attachments/assets/2703b8da-0c4f-44d8-90e5-feffc27387d5" alt="ROC Curve" width="300">
            <p align="center">ROC Curve for GoogleNet</p>
        </td>
        <td>
            <img src="https://github.com/user-attachments/assets/b5384c03-333b-4918-84f7-7dc9c8df928f" alt="Cross Validation" width="300">
            <p align="center">Cross Validation</p>
        </td>
        <td>
            <img src="https://github.com/user-attachments/assets/8a0b16ca-c661-476f-a75e-3800e8acb79d" alt="Lambda Variation" width="300">
            <p align="center">Lambda Variation</p>
        </td>
    </tr>
</table>
</div>

# Conclusion

<div  align = "justify">
Creating a predictive model for epilepsy using deep learning and continual learning is a big step forward in epilepsy management.
By using Googlenet as the transfer learning model we achieved the highest accuracy of 97% in training our dataset. 
Also, by using Elastic Weight Consolidation (EWC) as the continual learning technique with a learning rate scheduler our model enhanced the accuracy to 98.7%. Unlike older models that stay the same our continual learning model can learn from new information while still remembering what it has learnt before. This flexibility has made our model better at predicting seizures and giving timely alerts. 
</div>

# Credits

**1) Dataset link:** https://data.world/uci/epileptic-seizure-recognition
 
**2) Guide:**
Dr. Sandeep Kumar Satapathy, Professor, Vellore Institute of Technology, Chennai

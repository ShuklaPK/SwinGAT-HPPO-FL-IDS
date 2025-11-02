# SwinGAT-HPPO-FL-IDS: A Privacy-Preserving Intrusion Detection and Mitigation Framework for Software-Defined IoT Networks

## Overview  
This repository contains the source code, dataset configurations, and experimental results associated with the research paper titled **“SwinGAT-HPPO-FL: A Privacy-Preserving Intrusion Detection and Mitigation Framework for Software-Defined IoT Networks.”**  
The framework integrates **Swin Transformer**, **Graph Attention Network (GAT)**, **Hierarchical Proximal Policy Optimization (H-PPO)**, and **Federated Learning (FL)** to enable intelligent, adaptive, and privacy-preserving intrusion detection for **Software-Defined Internet of Things (SD-IoT)** environments.

---

## Methods Implemented

- **Caps-GJO** (Capsule Network with Golden Jackal Optimization) performs robust feature selection and abstraction.  
- **SwinGAT-Net** combines Swin Transformer and GAT to detect complex anomalies in SDN traffic.  
- **H-PPO** enables real-time mitigation with minimal latency.  
- **FL** ensures decentralized, privacy-preserving model updates across distributed nodes.  
- A **Flan-T5-based module** generates automated threat summaries and actionable feedback.  

Experiments conducted on the **Kaggle SDN-IoT dataset** demonstrate 98% detection accuracy, a 2% false-positive rate, and an average response latency below 120 ms, highlighting the model’s robustness, scalability, and real-time performance.

---

## Key Features  
- Multimodal SD-IoT data preprocessing  
- Hybrid Capsule Network + Golden Jackal Optimization feature selection  
- SwinGAT-Net: Transformer + Graph-based anomaly detection  
- Hierarchical Proximal Policy Optimization (H-PPO) for adaptive mitigation  
- Federated Learning-based secure, decentralized training  
- Automated threat summarization via Flan-T5  
- Comprehensive evaluation using accuracy, FPR, latency, and scalability metrics  

---



---

## Requirements  
- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- TensorFlow ≥ 2.11 (for FL components)  
- Transformers (Hugging Face)  
- Scikit-learn, NumPy, Pandas, Matplotlib  
- NetworkX, Ray, and Gym (for H-PPO implementation)  

To install dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset  
Experiments are performed on the **Kaggle SDN-IoT dataset**, which includes simulated and real-world SDN traffic instances labeled for multiple intrusion types.  
Dataset link: [[https://www.kaggle.com/datasets/hebadhirar/iot-sdn-ids-dataset](https://www.kaggle.com/datasets/hebadhirar/iot-sdn-ids-dataset)

---



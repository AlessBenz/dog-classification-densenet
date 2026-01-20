# Dog Breed Classification with DenseNet121

## Overview
This project investigates fine-grained dog breed classification using convolutional neural networks under limited data conditions.
We compare different training strategies for DenseNet121, trained **from scratch** and with **transfer learning**, focusing on overfitting behavior and generalization.

A key contribution of this work is the evaluation of a **two-phase (staged) training strategy** designed to mitigate overfitting when training deep models from scratch.

---

## Dataset
We use a curated subset of the **Stanford Dogs Dataset**, originally composed of 120 breeds.

To ensure sufficient samples per class, we retain only breeds with at least **200 images**, resulting in:
- **15 dog breeds**
- ~3,000 total images

The dataset is split into:
- **Training set**
- **Validation set**

No test split is used, as the focus of the project is on comparative analysis of training strategies rather than final deployment.

---

## Model Architecture
All experiments use **DenseNet121** as backbone.

A custom classification head is added, consisting of:
- Fully connected layer
- ReLU activation
- Dropout
- Final classification layer

Training uses:
- Cross-entropy loss
- Adam optimizer
- Data augmentation
- Early stopping based on validation accuracy

---

## Training Strategies

### 1. Scratch - Single Phase
The network is trained end-to-end from random initialization using a fixed learning rate.

### 2. Scratch - Two Phase (Staged Training)
Training is divided into two stages:
1. **Phase 1:** Full network training with higher learning rate.
2. **Phase 2:** Early convolutional layers are frozen and the learning rate is reduced.

This strategy aims to stabilize learned representations and reduce overfitting.

### 3. Transfer Learning (ImageNet)
DenseNet121 is initialized with **ImageNet-pretrained weights**.
Only the last convolutional block and the classifier are fine-tuned.

This experiment serves as a strong baseline and highlights the impact of pretrained representations.

---

## Quantitative Results

| Method | Top-1 Acc | Top-5 Acc | Macro F1 | LogLoss | Gap (Train–Val) |
|------|----------|-----------|----------|---------|-----------------|
| Scratch (1-phase) | 0.717 | 0.959 | 0.720 | 1.156 | 0.239 |
| Scratch (2-phase) | 0.782 | 0.962 | 0.782 | 0.938 | 0.200 |
| Transfer Learning | 0.944 | 0.999 | 0.944 | 0.198 | 0.048 |

The two-phase training strategy improves validation accuracy by approximately **6.5 percentage points** over standard scratch training and reduces the train–validation gap.
Transfer learning dramatically outperforms scratch training across all metrics.

---

## Discussion
Training DenseNet121 from scratch on a small fine-grained dataset leads to significant overfitting.
The proposed two-phase training strategy partially mitigates this issue, improving both validation accuracy and calibration.

However, transfer learning provides a substantial performance boost, demonstrating the importance of pretrained representations for fine-grained image classification tasks.

---

## Conclusion
This project shows that staged training is a viable strategy to improve generalization when training deep CNNs from scratch on limited data.
Nevertheless, transfer learning remains the most effective approach in this setting.

---

## Repository Structure
```text
.
├── src/                # Training and evaluation code
├── scripts/            # Dataset filtering and utilities
├── notebooks/          # Final analysis notebook
├── runs/               # Training outputs and checkpoints
└── README.md

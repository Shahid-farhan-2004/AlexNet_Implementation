# AlexNet on CIFAR-10 using PyTorch

## Overview

This project implements the **AlexNet Convolutional Neural Network (CNN)** using **PyTorch** and trains it on the **CIFAR-10** image classification dataset.

Since AlexNet expects an input size of **224 × 224**, the CIFAR-10 images (32 × 32) are resized before training.

---

## Dataset

**CIFAR-10** consists of:

- 60,000 color images
- 10 classes
- 50,000 training images
- 10,000 testing images

### Classes

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## Features

- AlexNet architecture
- CIFAR-10 dataset
- Image resizing to 224 × 224
- Data normalization
- Dropout regularization
- Adam optimizer
- CrossEntropy loss
- Model evaluation on the test dataset

---

## Project Structure

```
.
├── alexnet.py
├── README.md
└── data/
```

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision
```

---

## Model Architecture

### Feature Extractor

| Layer | Output Channels | Kernel | Stride | Padding |
|-------|----------------:|--------|--------|---------|
| Conv2D | 64 | 11×11 | 4 | 2 |
| MaxPool | - | 3×3 | 2 | - |
| Conv2D | 192 | 5×5 | 1 | 2 |
| MaxPool | - | 3×3 | 2 | - |
| Conv2D | 384 | 3×3 | 1 | 1 |
| Conv2D | 256 | 3×3 | 1 | 1 |
| Conv2D | 256 | 3×3 | 1 | 1 |
| MaxPool | - | 3×3 | 2 | - |

### Classifier

```
Dropout
↓
Linear(256×6×6 → 4096)
↓
ReLU
↓
Dropout
↓
Linear(4096 → 4096)
↓
ReLU
↓
Linear(4096 → 10)
```

---

## Data Preprocessing

```python
transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5))
])
```

---

## Training

Loss Function:

- CrossEntropyLoss

Optimizer:

- Adam

Learning Rate:

```
0.0005
```

Epochs:

```
20
```

Batch Size:

```
64
```

---

## Running the Project

```bash
python alexnet.py
```

---

## Evaluation

The model evaluates its performance using classification accuracy.

Example output:

```
Epoch 1 Loss: 1.67
Epoch 2 Loss: 1.34
...
Epoch 20 Loss: 0.42

Accuracy: 78.35%
```

---

## Technologies Used

- Python
- PyTorch
- Torchvision

---

## Future Improvements

- Train on GPU (CUDA)
- Data augmentation
- Learning rate scheduler
- Model checkpoint saving
- Confusion matrix visualization
- TensorBoard logging

---

## Author

**Shahid Farhan KP**

B.Tech Computer Science and Engineering

---

## License

This project is for educational purposes.

# AlexNet - PyTorch Reimplementation

A clean PyTorch implementation of **AlexNet** from the paper:

> Krizhevsky, Sutskever & Hinton (2012)
> _ImageNet Classification with Deep Convolutional Neural Networks._

This project was built from scratch as a learning exercise to understand how convolutional neural networks are implemented in PyTorch.

---

## Features

- AlexNet architecture implemented from scratch
- CIFAR-10 dataset support
- Training pipeline
- Evaluation pipeline
- Model checkpoint saving

---

## Project Structure

```
AlexNet/
│
├── model.py
├── train.py
├── evaluate.py
├── .gitignore
└── README.md
```

---

## Requirements

- Python 3.x
- PyTorch
- torchvision

Install dependencies:

```bash
pip install torch torchvision
```

---

## Training

```bash
python train.py
```

This will:

- Download CIFAR-10 (if necessary)
- Train AlexNet
- Save the trained weights as:

```
alexnet.pth
```

---

## Evaluation

```bash
python evaluate.py
```

This loads the saved weights and evaluates the model on the CIFAR-10 test set.

---

## Paper

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).

_ImageNet Classification with Deep Convolutional Neural Networks._

---

Educational project: This implementation was written from first principles to understand the architecture and training process rather than to maximize benchmark performance.

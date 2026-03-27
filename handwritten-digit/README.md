# handwritten-digit

Feedforward neural network trained on the QMNIST dataset to classify handwritten digits (0–9), achieving ~98% accuracy on the test set.

## What it does

- Loads 60,000 training images and 10,000 test images from QMNIST
- Trains a 3-layer fully connected network (784 → 128 → 64 → 10)
- Evaluates accuracy and validation loss after every epoch
- Reaches ~97–98% test accuracy in 10 epochs

## What I learnt

- **DataLoader**: how to wrap a `Dataset` for batched, shuffled loading
- **Training loop**: `model.train()` / `model.eval()` must be toggled each epoch; layers like Dropout and BatchNorm behave differently in each mode
- **Loss vs accuracy**: loss measures confidence, accuracy measures correctness — they can diverge. Val loss is a more reliable signal for detecting overfitting than accuracy alone
- **Overfitting**: when train loss keeps dropping but val loss rises, the model is memorising training-specific noise rather than learning general patterns
- **Early stopping**: save the model at each epoch and revert to the best val loss checkpoint
- **Optimizers**:
  - SGD: simple, one fixed learning rate, slower to converge but can find a better final minimum
  - Adam: adapts lr per weight, converges fast but can overfit in later epochs
  - AdamW: fixes Adam's weight decay bug, best default for most networks
  - Learning rate is not portable between optimizers (SGD: ~0.05, Adam/AdamW: ~0.001)
- **Reproducibility**: `torch.manual_seed()` is needed to compare runs fairly; without it results vary across runs due to random weight initialisation and batch shuffling

## Setup

Clone the QMNIST dataset repository next to this repository:

```
git clone https://github.com/facebookresearch/qmnist
```

## Run

```
cd handwritten-digit
uv run training_eval.py
```


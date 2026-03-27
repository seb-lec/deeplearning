# deeplearning projects

Personal repository to store my learnings about deep learning

## pytorch-1h

A guide to PyTorch in one hour by Sebastian Raschka (https://sebastianraschka.com/teaching/pytorch-1h/).
Python scripts used as notes.

## handwritten-digit

Exercise following the pytorch-1h guide, to recognise handwritten digits.


# deeplearning documentation

## Types of Neural Networks

| Type | Best For | Complexity |
|---|---|---|
| MLP | Tabular data | ⭐ |
| CNN | Images, video | ⭐⭐ |
| RNN | Short sequences | ⭐⭐ |
| LSTM / GRU | Long sequences | ⭐⭐⭐ |
| Transformer | Text, general purpose | ⭐⭐⭐ |
| Autoencoder | Compression, anomaly detection | ⭐⭐⭐ |
| VAE | Controlled generation | ⭐⭐⭐⭐ |
| GAN | Realistic generation | ⭐⭐⭐⭐ |
| GNN | Graph-structured data | ⭐⭐⭐⭐ |
| Diffusion | High-quality generation | ⭐⭐⭐⭐⭐ |

**After MLP, CNN and Transformer are the most valuable to learn next, as they dominate the field today.**

### 1. Multilayer Perceptron (MLP)
Fully connected layers where every neuron connects to every neuron in the next **layer**.
**Use when:** Data is tabular/structured (spreadsheets, databases), simple classification or regression tasks.

---

### 2. Convolutional Neural Network (CNN)
Uses filters that slide over data to detect local patterns, instead of connecting everything to everything.
**Use when:** Working with images or video. Very efficient because it exploits the spatial structure of pixels (nearby pixels are related).

---

### 3. Recurrent Neural Network (RNN)
Has a "memory" — the output of a **step** is fed back as input to the next **step**, making it aware of sequence order.
**Use when:** Sequential data like time series, text, or audio where **order matters**.

---

### 4. Long Short-Term Memory (LSTM) / Gated Recurrent Unit (GRU)
Improved versions of RNNs that solve the problem of RNNs "forgetting" things that happened far back in a sequence.
- **LSTM:** More powerful, more parameters
- **GRU:** Simpler, faster, often similar performance

**Use when:** Same as RNN but for longer sequences (e.g., long sentences, long time series).

---

### 5. Transformer
Replaces recurrence with an **attention mechanism** — instead of processing step by step, it looks at all parts of the input simultaneously and learns which parts to focus on.
**Use when:** Natural language (ChatGPT, BERT are transformers), but increasingly used for images, audio, and more. Currently the dominant architecture.

---

### 6. Autoencoder
Two parts: an **encoder** that compresses data into a smaller representation, and a **decoder** that reconstructs it.
**Use when:** Dimensionality reduction, denoising data, anomaly detection, or learning compact representations without labels (unsupervised).

---

### 7. Variational Autoencoder (VAE)
Like an autoencoder but the compressed representation is a **probability distribution**, which forces it to be smooth and structured.
**Use when:** Generating new data (new images, new samples) that look like your training data.

---

### 8. Generative Adversarial Network (GAN)
Two networks competing: a **Generator** tries to create fake data, a **Discriminator** tries to detect fakes. They improve each other.
**Use when:** Generating very realistic images, video, audio — or data augmentation.

---

### 9. Graph Neural Network (GNN)
Designed for data structured as **graphs** (nodes and edges), where the standard grid assumption of CNNs doesn't apply.
**Use when:** Social networks, molecule property prediction, recommendation systems, knowledge graphs.

---

### 10. Diffusion Model
Learns to **gradually denoise** random noise back into structured data. Currently state of the art for image generation.
**Use when:** High-quality image/audio/video generation (Stable Diffusion, DALL·E 3 use this).

---

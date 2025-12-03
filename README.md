
# Conditional VAE-GAN Framework for Controlled Image Synthesis using CIFAR-100

This project implements and compares two generative models on the CIFAR-100 dataset:
```
	1.	Conditional Variational Autoencoder (cVAE) – a baseline model that learns a structured latent representation and reconstructs images.
	2.	Conditional VAE–GAN (CVAE-GAN) – an enhanced hybrid model that improves sample sharpness and class-specific generation using adversarial training.
```
The goal is to analyze how adding a discriminator affects:

	•	reconstruction quality
	•	sample sharpness
	•	label-controlled generation
	•	latent space continuity
	•	training stability


---

## Directory Structure
```
cvae-gan-cifar/
│
├── data/                        # CIFAR-100 dataset (downloaded automatically)
│
├── plots/                       # Loss curves (VAE + CVAE-GAN)
│
├── eval_outputs/                # CVAE-GAN recon, samples, interpolation
├── eval_outputs_vae/            # VAE recon, samples, interpolation
│
├── src/
│   ├── dataset.py               # CIFAR-100 dataloaders
│   ├── train.py                 # CVAE-GAN training script
│   ├── train_vae.py             # VAE baseline training script
│   ├── evaluate.py              # CVAE-GAN evaluation
│   ├── evaluate_vae.py          # VAE evaluation
│   ├── models/                  # Encoder, Generator, Discriminator
│   └── utils/                   # losses, device helpers
│
├── checkpoint_final.pt          # CVAE-GAN final checkpoint
├── checkpoint_vae.pt            # VAE baseline checkpoint
│
├── requirements.txt
├── Sample-Report-VAE.txt
├── Sample-Report-CVAE-GAN.txt
└── README.md
```

---

## Installation

1. Clone repo
```
git clone https://github.com/aryaa-deshpande/cvae-gan-cifar.git
cd cvae-gan-cifar
```
2. Create virtual environment
```
python3.10 -m venv venv
source venv/bin/activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
Works on:
	•	Apple Silicon (MPS)
	•	CPU
	•	CUDA (if available)

---

## Models Overview

### 1. Conditional VAE (Baseline)
	•	Encoder outputs mu, logvar
	•	Reparameterization → latent vector z
	•	Decoder reconstructs image conditioned on class label
	•	Loss: recon + KL
	•	Pros: smooth latent space, low recon error
	•	Cons: blurry samples


### 2. Conditional VAE–GAN (Hybrid)
	•	Uses same VAE encoder & decoder
	•	Adds discriminator for sharper images
	•	Loss: recon + KL + GAN
	•	Pros: sharper samples, better class control
	•	Cons: higher recon error, training oscillations

---

## Training

### 1. Train the VAE Baseline
```
python3 src/train_vae.py
```
Outputs:

	•	checkpoint_vae.pt
	•	plots/losses_vae.png
	•	sample grids inside eval_outputs_vae/

---

### 2. Train the CVAE–GAN (Hybrid Model)
```
python3 src/train.py
```
Outputs:

	•	checkpoint_final.pt
	•	plots/losses.png
	•	epoch-wise samples in samples/
	•	evaluation outputs in eval_outputs/

---

## Evaluation

### 1. Evaluate VAE
```
python3 src/evaluate_vae.py
```
Generates:

	•	reconstructions_vae.png
	•	class_samples_0to7_vae.png
	•	latent_interpolation_vae.png
	•	prints Reconstruction MSE

---

### 2. Evaluate CVAE–GAN
```
python3 src/evaluate.py
```
Generates:

	•	reconstructions.png
	•	class_samples_0to7.png
	•	latent_interpolation.png
	•	prints Reconstruction MSE

---

## Results Summary

### VAE Baseline

	•	Recon MSE: ~0.2033
	•	Smooth latent interpolation
	•	Good reconstructions (blurry but structured)
	•	Weak class-conditioned samples
	•	Extremely stable training

### CVAE–GAN

	•	Recon MSE: ~0.41 (expected)
	•	Sharper, more class-specific samples
	•	Richer textures
	•	Slight adversarial oscillations (normal)
	•	Good latent interpolation

### High-level takeaway:
```
VAE = learns representation

CVAE–GAN = learns sharper generation
```
Together, they provide a complete generative modeling pipeline.

---

## Reproducibility Checklist

	1.	Run train_vae.py → get baseline
	2.	Run evaluate_vae.py → get recon + samples + interpolation
	3.	Run train.py → train hybrid model
	4.	Run evaluate.py → evaluate hybrid model
	5.	Compare loss curves
	6.	Compare reconstructions
	7.	Compare generated samples
	8.	Include figures in report

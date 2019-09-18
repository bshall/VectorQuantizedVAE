# Vector Quantized VAE
A PyTorch implementation of [Continuous Relaxation Training of Discrete Latent Variable Image Models](http://bayesiandeeplearning.org/2017/papers/54.pdf).

Ensure you have Python 3.7 and PyTorch 1.2 or greater. 
To train the `VQVAE` model with 8 categorical dimensions and 128 codes per dimension 
run the following command:
```
python train.py --model=VQVAE --latent-dim=8 --num-embeddings=128
``` 
To train the `GS-Soft` model use `--model=GSSOFT`.

<p align="center">
    <img src="assets/reconstructions.png?raw=true" alt="VQVAE Reconstructions">
</p>

The `VQVAE` model gets ~4.82 bpd while the `GS-soft` model gets ~4.6 bpd.

# Analysis of the Codebooks 

As demonstrated in the paper, the codebook matrices are low-dimensional, spanning only a few dimensions:

<p align="center">
    <img src="assets/variance_ratio.png?raw=true" alt="Explained Variance Ratio">
</p>

Projecting the codes onto the first 3 principal components shows that the codes typically tile 
continuous 1- or 2-D manifolds:

<p align="center">
    <img src="assets/codebooks.png?raw=true" alt="Codebook principal components">
</p>

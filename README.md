# DCGNN

This is a PyTorch implementation of the paper: “Causality Meets Temporality: Inferring Latent Interactions of Entities from Pattern Prototypes for Stock Prediction.”


## Requirements
* python==3.7.13
* torch==1.8.0
* talib==0.4.19
* numpy==1.21.6
* pandas==1.3.5


## How to train the model
1. Run `pattern_prototype_recognition.ipynb`.
This script would recognize pattern prototypes of all directed stock pairs.
2. Run `PA-TMM.ipynb`.
This script would build a DCGNN model, and then train and evaluate it.

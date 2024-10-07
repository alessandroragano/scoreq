# SCOREQ: Speech Contrastive Regression for Quality Assessment

SCOREQ is a framework for speech quality assessment based on pre-training the encoder with the SCOREQ loss function.

This repo provides four speech quality metrics trained with the SCOREQ framework.

| Domain |  Usage Mode | 
|---|---|
| Natural speech   | No-reference 
| Natural speech   | Non-matching reference, full-reference
| Synthetic speech | No-reference 
| Synthetic speech | Non-matching reference


## Installation
SCOREQ is hosted on PyPi. It can be installed in your Python environment with the following command
```
pip install scoreq
```

The model works with 16 kHz sampling rate. If your sampling rate is different, automatic resampling is performed.
SCOREQ was built with Python 3.9.16.

## Using SCOREQ

Reference files can be any clean speech.

### Using SCOREQ from the command line

### Using SCOREQ inside Python

## NeurIPS 2024
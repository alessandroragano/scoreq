# SCOREQ: Speech Contrastive Regression for Quality Assessment

SCOREQ is a framework for speech quality assessment based on pre-training the encoder with the SCOREQ loss.

This repo provides four speech quality metrics trained with the SCOREQ framework.

| Domain | Train Set | Usage Mode | Prediction
|---|---|---|---|
| Natural speech   | NISQA TRAIN SIM |No-reference | Mean Opinion Score
| Natural speech   | NISQA TRAIN SIM |Non-matching reference, full-reference | Euclidean distance clean speech
| Synthetic speech | VoiceMOS 22 Train Set |No-reference | Mean Opinion Score
| Synthetic speech | VoiceMOS 22 Train Set | Non-matching reference | Euclidean distance clean speech

## Performance - Pearson Correlation
Performance is calculated by averaging per condition where available.

| Dataset           |   Domain   | NISQA (NISQA TRAIN SIM) | NR-PESQ | NR-SI SDR | NORESQA-M | NR-SCOREQ Natural| NR-SCOREQ Synthetic |
|-------------------|------|-------|---------|-----------|-----------|------------------| --------------------|
| NISQA TEST FOR    |  Online Conferencing Simulated (codecs, background noise, packet loss, etc. )    | 0.91   | 0.79      | 0.74   |   0.68    |    **0.97**      |   0.82
| NISQA TEST P501   |  Online Conferencing Simulated (codecs, background noise, packet loss, etc. )   | 0.94   | 0.88      | 0.81   |   0.70    |    **0.96**      |   0.86
| VoiceMOS Test 1  |   Speech Synthesis    | 0.54   | 0.71      | 0.67   |   0.85    |       0.86       |  **0.90**
| VoiceMOS Test 2  |   Speech Synthesis   |  0.64   | 0.49      | 0.55   |   0.91    |      0.82        |  **0.98**
| NOIZEUS          |    Speech Enhancement, Background Noise |  0.85   | 0.75      | 0.70   |   0.15    |    **0.91**      |   0.59
| NISQA TEST LT     |   Online Conferencing Live   | 0.84   | 0.66      | 0.56   | 0.60      |    **0.86**      |   0.81
| P23 EXP3          |   Packet Loss, Codecs   | 0.82   | 0.77      | 0.17   | 0.71      |    **0.94**      |   0.88
| TCD VOIP         |    VoIP Degradations  |  0.76   | 0.76      | 0.76   | 0.61      |    0.85      |  **0.87**
| TENCENT           |   Online Conferencing Simulated (codecs, background noise, packet loss, etc. )  | 0.78   | 0.78      | 0.77   | 0.57      |    **0.86**      |   0.78
| P23 EXP1          |   Codecs   | 0.76   | 0.70      | 0.82   | 0.40      |    **0.96**      |   0.92
| TENCENT-Rev              |   Real-World Reverberation   | 0.40   | 0.36      | 0.32   | 0.36      |    **0.79**      |   0.43


## Installation

**Requirement:** Python 3.10 or newer.

SCOREQ is hosted on PyPi and uses ONNX Runtime inference by default. This removes the need for heavy dependencies like `fairseq` for standard use.

Choose the installation method that best suits your needs:

### Standard Installation (CPU)

For most users. This provides a CPU-based installation for inference.
```bash
pip install scoreq
```
### Fast Inference (GPU)
For users with a compatible NVIDIA GPU and CUDA setup, install the [gpu] extra for faster inference.
```bash
pip install scoreq[gpu]
```
### PyTorch/Fairseq version
For users who wish to fine-tune models or work with the original PyTorch framework, install the [pytorch] extra. This will install fairseq, torch, and all other development dependencies.
```bash
pip install scoreq[pytorch]
```


### First run
The PyTorch weights are hosted on Zenodo. The first run might be slower due to model download. 

## Using SCOREQ 
The expected sampling rate is 16 kHz. The script automatically resamples audio with different sampling rates. 
SCOREQ models accept variable input length.

SCOREQ can be used in 2 modes and for 2 domains by setting the arguments ```data_domain``` and ```mode```.

### Using SCOREQ from the command line



| Domain |  Usage Mode | CLI 
|---|---|---|
| Natural speech   | No-reference | ```python -m scoreq data_domain natural mode nr /path/to/test_audio ```
| Natural speech   | Non-matching reference, full-reference | ```python -m scoreq data_domain natural mode ref /path/to/test_audio --ref_path /path/to/ref_audio```
| Synthetic speech | No-reference |```python -m scoreq data_domain synthetic mode nr /path/to/test_audio ```
| Synthetic speech | Non-matching reference | ```python -m scoreq data_domain synthetic mode ref /path/to/test_audio --ref_path /path/to/ref_audio```


### Using SCOREQ inside Python
Inside python you first need to import the package.
Examples using wav files provided in the data directory.

```
import scoreq

# Predict quality of natural speech in NR mode
nr_scoreq = scoreq.Scoreq(data_domain='natural', mode='nr')
pred_mos = nr_scoreq.predict(test_path='./data/opus.wav', ref_path=None)

# Predict quality of natural speech in REF mode
ref_scoreq = scoreq.Scoreq(data_domain='natural', mode='ref')
pred_distance = ref_scoreq.predict(test_path='./data/opus.wav', ref_path='./data/ref.wav')

# Predict quality of synthetic speech in NR mode
nr_scoreq = scoreq.Scoreq(data_domain='synthetic', mode='nr')
pred_mos = nr_scoreq.predict(test_path='./data/opus.wav', ref_path=None)

# Predict quality of synthetic speech in REF mode
ref_scoreq = scoreq.Scoreq(data_domain='synthetic', mode='ref')
pred_distance = ref_scoreq.predict(test_path='./data/opus.wav', ref_path='./data/ref.wav')
```

## Correct usage

### ```data_domain```

This parameter selects the underlying pre-trained model. Choosing the correct domain is critical for obtaining meaningful results.

| ```data_domain``` |  Recommended For | Description 
|---|---|---|
| natural   | Audio Codecs, VoIP, Telephony, Speech Enhancement, Audio Restoration | Use for audio that originates from recordings of human speakers.
| synthetic   | Text-to-Speech (TTS), Voice Conversion (VC), Generative Speech Models | Use for audio that has been synthetically generated by a machine.

### ```mode```
This parameter determines the operational mode of the model, specifying whether a reference audio signal is used for the assessment.

`mode=nr` (No-Reference): This mode assesses the quality of the input audio without requiring a reference signal. It is suitable for any application where a "clean" source is unavailable. This is common for evaluating synthetic domain outputs (e.g., Text-to-Speech) but is also useful for natural speech applications in blind scenarios.

`mode=ref` (Reference-based): This mode uses a reference audio signal to evaluate the test audio. It can operate in two ways, depending on the reference provided.

- **Full-Reference**: The metric operates in this sub-mode when the provided reference audio is the clean counterpart of the test audio. This is the recommended approach for applications in the natural speech domain where the original source is typically available (e.g., evaluating speech codecs).

- **Non-Matching Reference**: The metric operates in this sub-mode if a random, clean speech sample (that is not the direct source of the test audio) is used as the reference. SCOREQ learns a distance metric and expects clean speech for this purpose. Note: The model has not been evaluated with other types of non-matching references.


### Input Length

SCOREQ accepts inputs of any length. However, it was trained on segments up to 4 seconds and evaluated on segments up to 15 seconds. We recommend trimming your audio to 10–15 seconds to avoid running out of memory with longer recordings. This duration is sufficient for accurate audio quality predictions.

### Sampling rate
SCOREQ automatically resamples input files to 16 kHz, the selected sampling rate for training. The model was evaluated on four test sets in full-band mode (48 kHz), yielding promising results. While we encourage experimenting with SCOREQ at higher sampling rates, any conclusions drawn from using rates above 16 kHz should be validated through listening tests.

## Paper - NeurIPS 2024
Check our paper [here](https://proceedings.neurips.cc/paper_files/paper/2024/file/bece7e02455a628b770e49fcfa791147-Paper-Conference.pdf)
```
@inproceedings{ragano2024scoreq,
 author = {Ragano, Alessandro and Skoglund, Jan and Hines, Andrew},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {105702--105729},
 publisher = {Curran Associates, Inc.},
 title = {SCOREQ: Speech Quality Assessment with Contrastive Regression},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/bece7e02455a628b770e49fcfa791147-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}


```
[![DOI](https://zenodo.org/badge/868888288.svg)](https://doi.org/10.5281/zenodo.14735580)

The SCOREQ code is licensed under MIT license. Dependencies of the project are available under separate license terms.

Copyright © 2024 Alessandro Ragano

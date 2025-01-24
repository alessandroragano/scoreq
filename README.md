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
<!---
![pc](https://raw.githubusercontent.com/alessandroragano/scoreq/main/figs/results.png)
-->

| Dataset           |   Domain   | NISQA (NISQA TRAIN SIM) | NR-PESQ | NR-SI SDR | NORESQA-M | NR-SCOREQ Natural| NR-SCOREQ Synthetic |
|-------------------|------|-------|---------|-----------|-----------|------------------| --------------------|
| NISQA TEST FOR    |  Online Conferencing Simulated (codecs, background noise, packet loss, etc. )    | 0.91   | 0.79      | 0.74   |   0.68    |    **0.97**      |   0.82
| NISQA TEST P501   |  Online Conferencing Simulated (codecs, background noise, packet loss, etc. )   | 0.94   | 0.88      | 0.81   |   0.70    |    **0.96**      |   0.86
| DNS Squim         |  Speech Enhancement, Background Noise    | //     | 0.96      | 0.99   |     //    |       //         |   //
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
SCOREQ is hosted on PyPi. It can be installed in your Python environment with the following command
```
pip install scoreq
```

The expected sampling rate is 16 kHz. The script automatically resamples audio with different sampling rates. 
SCOREQ models accept variable input length.

### First run
The PyTorch weights are hosted on Zenodo. The first run might be slower due to model download. 

## Using SCOREQ 
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

### Reference Mode

Use `mode=ref` for both non-matching reference and full-reference scenarios, depending on the clean speech input:

- The metric operates in **full-reference mode** when provided with the clean counterpart of the audio.
- It operates in **non-matching reference mode** if any random clean speech is used as input.

Full-reference mode is expected for applications related to the natural speech domain where the clean counterpart is typically available e.g., speech codecs. 
SCOREQ learns a distance metric and expects clean speech as the non-matching reference. Note that the model has not been evaluated with other types of non-matching references.

### Input Length

SCOREQ accepts inputs of any length. However, it was trained and evaluated on segments up to 15 seconds. We recommend trimming your audio to 10–15 seconds to avoid running out of memory with longer recordings. This duration is sufficient for accurate audio quality predictions.

### Sampling rate
SCOREQ automatically resamples input files to 16 kHz, the selected sampling rate for training. The model was evaluated on four test sets in full-band mode (48 kHz), yielding promising results. While we encourage experimenting with SCOREQ at higher sampling rates, any conclusions drawn from using rates above 16 kHz should be validated through listening tests.

## Paper - NeurIPS 2024
Check our paper [here](https://arxiv.org/pdf/2410.06675)
```
@article{ragano2024scoreq,
  title={SCOREQ: Speech Quality Assessment with Contrastive Regression},
  author={Ragano, Alessandro and Skoglund, Jan and Hines, Andrew},
  journal={arXiv preprint arXiv:2410.06675},
  year={2024}
}
```
[![DOI](https://zenodo.org/badge/868888288.svg)](https://doi.org/10.5281/zenodo.14735580)

The SCOREQ code is licensed under MIT license. Dependencies of the project are available under separate license terms.

Copyright © 2024 Alessandro Ragano

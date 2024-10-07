# SCOREQ: Speech Contrastive Regression for Quality Assessment

SCOREQ is a framework for speech quality assessment based on pre-training the encoder with the SCOREQ loss.

This repo provides four speech quality metrics trained with the SCOREQ framework.

| Domain |  Usage Mode | Prediction
|---|---|---|
| Natural speech   | No-reference | Mean Opinion Score
| Natural speech   | Non-matching reference, full-reference | Euclidean distance clean speech
| Synthetic speech | No-reference | Mean Opinion Score
| Synthetic speech | Non-matching reference | Euclidean distance clean speech



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

## Other
We provide the best model for each domain-mode pair. 

Use ```mode=ref``` for both non-matching reference or full-reference. This is affected by the clean speech used as input.

If you pass the clean counterpart, the metric will run in full-reference mode. 
If you pass any clean speech, the metric will run in non-matching reference mode.

Full-reference mode is expected to be used only for natural speech, where the clean copy is available.

SCOREQ learns a distance and it expects clean speech as non-matching reference. 
The model has not been evaluated for other non-matching references.

## Paper (available soon)

The SCOREQ code is licensed under MIT license. Dependencies of the project are available under separate license terms.

Copyright © 2024 Alessandro Ragano
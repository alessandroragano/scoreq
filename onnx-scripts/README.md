# Developer Scripts

This directory contains scripts used for building and verifying model weights. These are not required for using the `scoreq` package.

- `export_to_onnx.py`: Exports the original PyTorch models to the ONNX format. This is used to generate the released models.
- `verify_onnx_batch.py`: Verifies the numerical correctness of the exported ONNX models against the original PyTorch models by running a test suite defined in a CSV file. This script loads a csv of test sets to verify that predictions of the ONNX models match the fairseq original version of SCOREQ.
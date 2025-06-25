import torch
import numpy as np
import onnxruntime as ort
import scoreq
import argparse
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import math
import torch.nn.functional as F

# The wav2vec 2.0 model's CNN feature extractor has a total stride of 320
PADDING_MULTIPLE = 320

def dynamic_pad(x, multiple=PADDING_MULTIPLE, dim=-1, value=0):
    """Pads the input tensor to be a multiple of PADDING_MULTIPLE."""
    tsz = x.size(dim)
    required_len = math.ceil(tsz / multiple) * multiple
    remainder = required_len - tsz
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(x, pad_offset + (0, remainder), value=value)

def verify_model_for_db(domain, mode, db_name, db_df, root_dir):
    """
    Verifies a specific model combination against a single database.
    """
    print(f"\n===== Verifying: domain='{domain}', mode='{mode}' on database='{db_name}' =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    providers = ['CUDAExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
    
    scoreq_instance = scoreq.Scoreq(data_domain=domain, mode=mode, device=device.type)
    original_model = scoreq_instance.model

    onnx_filename = f"scoreq_{domain}_{mode}.onnx"
    ort_session = ort.InferenceSession(onnx_filename, providers=providers)
    input_name = ort_session.get_inputs()[0].name
    print(f"Using PyTorch device: {device} | ONNX provider: {ort_session.get_providers()[0]}")

    original_scores, onnx_scores = [], []
    for _, row in tqdm(db_df.iterrows(), total=db_df.shape[0], desc=f"DB: {db_name}"):
        test_audio_path = os.path.join(root_dir, row['filepath'])
        
        path_parts = test_audio_path.split(os.sep)
        try:
            # Find the index of the 'deg' directory
            deg_index = path_parts.index('deg')
            # Replace it with 'ref'
            path_parts[deg_index] = 'ref'
            
            # Reconstruct the directory path
            ref_dir = os.sep.join(path_parts[:-1]) # Join all parts except the filename
            
            # Construct the reference filename
            deg_filename = os.path.basename(test_audio_path)
            base_filename = deg_filename.split('_', 1)[1]
            ref_audio_path = os.path.join(ref_dir, base_filename)
        except (ValueError, IndexError):
            # If 'deg' is not in the path or filename format is wrong, skip in ref mode
            if mode == 'ref': continue
            ref_audio_path = "" # Set to empty if not in ref mode

        if not os.path.exists(test_audio_path) or (mode == 'ref' and not os.path.exists(ref_audio_path)):
            continue

        test_wave_raw = scoreq_instance.load_processing(test_audio_path)
        test_wave_padded = dynamic_pad(test_wave_raw).to(device)

        with torch.no_grad():
            if mode == 'nr':
                original_score = original_model(test_wave_padded).item()
            else:
                ref_wave_raw = scoreq_instance.load_processing(ref_audio_path)
                ref_wave_padded = dynamic_pad(ref_wave_raw).to(device)
                test_emb = original_model(test_wave_padded)
                ref_emb = original_model(ref_wave_padded)
                original_score = torch.cdist(test_emb, ref_emb).item()

        test_numpy = test_wave_padded.cpu().numpy()
        if mode == 'nr':
            onnx_score = ort_session.run(None, {input_name: test_numpy})[0].item()
        else:
            ref_numpy = ref_wave_padded.cpu().numpy()
            test_emb_onnx = ort_session.run(None, {input_name: test_numpy})[0]
            ref_emb_onnx = ort_session.run(None, {input_name: ref_numpy})[0]
            onnx_score = np.linalg.norm(test_emb_onnx - ref_emb_onnx)
        
        original_scores.append(round(original_score, 5))
        onnx_scores.append(round(onnx_score, 5))

    if not original_scores:
        print(f"\nCould not find any valid audio files for database '{db_name}'. Please check paths and filename logic.")
        return

    mae = mean_absolute_error(original_scores, onnx_scores)
    
    print(f"\n--- Results for database: '{db_name}' ---")
    print(f"  Model: {domain}/{mode}")
    print(f"  Processed {len(original_scores)} files.")
    print(f"  Mean Absolute Error (MAE) after rounding: {mae:.8f}")
    
    if mae < 1e-5:
        print("PASSED: models are identical!")
    else:
        print("FAILURE: models are divergent.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final verification of ONNX models on GPU, grouped by database.")
    parser.add_argument('--csv-file', type=str, default='scoreq_test.csv', help="Path to the input CSV file.")
    parser.add_argument('--root', type=str, default='.', help="Root directory where datasets are located.")
    args = parser.parse_args()

    try:
        dataframe = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at '{args.csv_file}'")
        exit()

    model_combinations = [{'domain': 'natural', 'mode': 'nr'}, {'domain': 'natural', 'mode': 'ref'},
                          {'domain': 'synthetic', 'mode': 'nr'}, {'domain': 'synthetic', 'mode': 'ref'}]

    for db_name, db_df in dataframe.groupby('db'):
        for combo in model_combinations:
            verify_model_for_db(
                domain=combo['domain'], 
                mode=combo['mode'], 
                db_name=db_name,
                db_df=db_df,
                root_dir=args.root
            )
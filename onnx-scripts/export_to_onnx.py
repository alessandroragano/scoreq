import torch
import torch.nn.functional as F
import fairseq.models.wav2vec.wav2vec2 as wav2vec2
import scoreq
import os

def patch_fairseq_for_onnx():
    """
    Applies a monkey-patch to a fairseq function to make it fully ONNX-traceable.
    """
    if getattr(patch_fairseq_for_onnx, 'patched', False):
        return
        
    print("Applying DYNAMIC monkey-patch to fairseq's 'pad_to_multiple' function...")
    
    def dynamic_pad_to_multiple(x, multiple, dim=-1, value=0):
        # Convert size to a tensor to use torch.* functions
        tsz = torch.tensor(float(x.size(dim)), device=x.device)

        # Calculate required padding using only traceable torch operations
        # This avoids Python-native branching (.item())
        m = tsz / multiple
        required_len = torch.ceil(m) * multiple
        remainder = (required_len - tsz).to(torch.long) # F.pad needs integer padding

        # F.pad with a remainder of 0 is a no-op.
        pad_offset = (0,) * (-1 - dim) * 2
        return F.pad(x, pad_offset + (0, remainder), value=value), remainder

    # Apply patch
    wav2vec2.pad_to_multiple = dynamic_pad_to_multiple
    patch_fairseq_for_onnx.patched = True # Mark as patched
    print("Patch applied successfully.")

def export_model(domain, mode):
    """Loads a scoreq model for a specific domain/mode and exports it to ONNX."""
    
    print(f"\n--- Processing Model: domain='{domain}', mode='{mode}' ---")
    
    print("Loading Scoreq model...")
    scoreq_instance = scoreq.Scoreq(data_domain=domain, mode=mode)
    model = scoreq_instance.model
    model.eval()
    print("Model loaded successfully.")

    print("Moving model to CPU for ONNX export...")
    model.to("cpu")
    patch_fairseq_for_onnx()

    dummy_input = torch.randn(1, 16001) # Use a non-multiple length (if 16000 this will make wrong predictions)
    output_filename = f"scoreq_{domain}_{mode}.onnx"
    print(f"Starting ONNX export to '{output_filename}'...")

    torch.onnx.export(
        model,
        dummy_input,
        output_filename,
        export_params=True,
        do_constant_folding=True,
        input_names=["audio_input"],
        output_names=["mos_output"],
        dynamic_axes={
            "audio_input": {0: "batch_size", 1: "samples"},
            "mos_output": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"Successfully exported model to '{output_filename}'")
    print(f"File size: {os.path.getsize(output_filename) / 1e6:.2f} MB")


if __name__ == "__main__":
    model_combinations = [
        {'domain': 'natural', 'mode': 'nr'},
        {'domain': 'natural', 'mode': 'ref'},
        {'domain': 'synthetic', 'mode': 'nr'},
        {'domain': 'synthetic', 'mode': 'ref'},
    ]

    print("Starting batch export with DYNAMIC padding...")
    for combo in model_combinations:
        export_model(domain=combo['domain'], mode=combo['mode'])
    
    print("\nBatch export process completed.")
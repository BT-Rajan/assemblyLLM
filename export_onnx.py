# export_onnx.py
import torch
import os
import json
from config import AssemblyRouterConfig
from model import AssemblyRouterLM

def export_to_onnx():
    print("Initializing Unified ONNX structural conversion...")
    
    vocab_path = "data/tokenizer_vocab.json"
    if not os.path.exists(vocab_path):
        print("Error: Missing tokenizer assets.")
        return
        
    with open(vocab_path, "r") as f:
        v_data = json.load(f)
        vocab_size = len(v_data["stoi"])

    cfg = AssemblyRouterConfig(vocab_size=vocab_size)
    model = AssemblyRouterLM(cfg)
    
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print("Error: Trained model weights not found.")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # Create dummy placeholder matching your exact max sequence shape
    dummy_input = torch.zeros((1, cfg.max_seq_len), dtype=torch.long)
    
    os.makedirs("docs", exist_ok=True)
    onnx_output_path = "docs/assembly_router.onnx"

    # Clean up old segmented files if they exist to prevent browser confusion
    if os.path.exists(onnx_output_path + ".data"):
        os.remove(onnx_output_path + ".data")

    # Exporting a unified, single binary block
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        # REMOVED complex dynamic axes shapes to force deep serialization inside one file boundary
    )
    
    print(f"✓ Success! Unified ONNX Model compiled to a single file: {onnx_output_path}")
    print(f"Current file size should match your complete ~57MB network framework footprint.")

if __name__ == "__main__":
    export_to_onnx()
# train.py
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import AssemblyRouterConfig, AssemblyTrainConfig
from model import AssemblyRouterLM  # Standalone network independent of other projects
from dataset import AssemblyTokenizer, AssemblyDataset, assembly_collate_fn

def train_assembly_router():
    # 1. INITIALIZE SYSTEM HARDWARE ACCELERATION
    device = "cpu"
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    print(f"Executing Deep Learning Engine optimization via hardware target: {device}")

    # 2. LOAD INFRASTRUCTURE CONFIGURATIONS
    tokenizer = AssemblyTokenizer()
    model_cfg = AssemblyRouterConfig(vocab_size=tokenizer.vocab_size)
    train_cfg = AssemblyTrainConfig()
    
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    # 3. CONSTRUCT DATA LOADING PIPELINE 
    train_ds = AssemblyDataset("data/train.jsonl", tokenizer, model_cfg.max_seq_len)
    eval_ds = AssemblyDataset("data/eval.jsonl", tokenizer, model_cfg.max_seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=assembly_collate_fn)
    eval_loader = DataLoader(eval_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=assembly_collate_fn)

    # 4. INSTANTIATE MODEL ARCHITECTURE WEIGHTS
    # Utilizing your clean, scratch-built Transformer Decoder layers
    model = AssemblyRouterLM(model_cfg)
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_cfg.learning_rate, 
        weight_decay=train_cfg.weight_decay, 
        betas=(train_cfg.beta1, train_cfg.beta2)
    )

    # 5. EXECUTE CORE ITERATIVE TRAINING STEPS
    step = 0
    epochs = math.ceil(train_cfg.max_steps / len(train_loader))
    best_val_loss = float("inf")
    
    print(f"Beginning assembly matrix optimization across {epochs} programmatic epochs.")
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            if step >= train_cfg.max_steps: break
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Linear Cosine Learning Rate Schedule Adjustment
            lr = train_cfg.learning_rate
            if step < train_cfg.warmup_steps:
                lr *= (step + 1) / train_cfg.warmup_steps
            else:
                progress = (step - train_cfg.warmup_steps) / (train_cfg.max_steps - train_cfg.warmup_steps)
                lr *= 0.5 * (1.0 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups: param_group['lr'] = lr
            
            # Forward optimization step execution
            optimizer.zero_grad()
            logits, loss = model(inputs, labels)
            loss.backward()
            
            # Prevent gradient explosion calculations via clipping thresholds
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Step Validation Checking Loop
            if step % train_cfg.eval_interval == 0:
                model.eval()
                val_loss_accum = 0.0
                with torch.no_grad():
                    for v_inputs, v_labels in eval_loader:
                        v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
                        _, v_loss = model(v_inputs, v_labels)
                        val_loss_accum += v_loss.item()
                
                avg_val_loss = val_loss_accum / len(eval_loader)
                print(f"Step {step:04d} | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f} | LR: {lr:.6f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), os.path.join(train_cfg.checkpoint_dir, "best_model.pt"))
                    
                model.train()
                
            step += 1

    # Save out the structural vocabulary schema mapping for production inference pipelines
    with open("data/tokenizer_vocab.json", "w") as f:
        json.dump({"stoi": tokenizer.stoi, "itos": tokenizer.itos}, f)
        
    print(f"Training finalized. System weights serialized safely to {train_cfg.checkpoint_dir}/best_model.pt")

if __name__ == "__main__":
    train_assembly_router()
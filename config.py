# config.py
from dataclasses import dataclass

@dataclass
class AssemblyRouterConfig:
    """Defines the core deep learning architecture shapes tailored 

    specifically for parsing alphanumeric code variables.
    """
    vocab_size: int = 1024       # Compact vocabulary constraint to force explicit token-binding 
    max_seq_len: int = 256       # Comfortably fits multi-part diagnostic engineering inquiries
    
    n_layer: int = 8             # Expanded depth to accurately capture structural multi-hop cross-referencing
    n_embd: int = 384            # Latent space dimension keeping model computation incredibly fast
    n_head: int = 8              # 8 attention heads tracking interlocking component interfaces
    
    dropout: float = 0.05        # Keeps the network heavily focused on strict grammar layouts
    bias: bool = True            # Preserves structural bias variations across linear matrices

@dataclass
class AssemblyTrainConfig:
    """Hyperparameters tuned for rapid optimization over low-entropy token targets."""
    
    batch_size: int = 16         # Small, stable batch processing profile 
    learning_rate: float = 5e-4  # Maximum learning rate ceiling for the Cosine optimization curve
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    
    max_steps: int = 1500        # Training steps required to safely lock down pattern memorization
    warmup_steps: int = 150      # Gradual optimization ramp up to shield token weights early on
    
    eval_interval: int = 100     # Interval steps before validating against cross-validation data splits
    eval_iters: int = 20         
    checkpoint_dir: str = "checkpoints"
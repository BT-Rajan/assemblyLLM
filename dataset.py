# dataset.py
import torch
from torch.utils.data import Dataset
import json

class AssemblyTokenizer:
    """A highly deterministic, localized character-level tokenizer featuring

    explicit vocabulary binding safety for alphanumeric codes.
    """
    def __init__(self):
        # Establish structural special characters
        self.chars = sorted(list(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ {}:\",[]?.!=-/()@\n'")))
        self.stoi = {ch: i + 4 for i, ch in enumerate(self.chars)}
        
        # Explicit special boundaries
        self.stoi["<pad>"] = 0
        self.stoi["<unk>"] = 1
        self.stoi["<s>"] = 2
        self.stoi["</s>"] = 3
        
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> list:
        # Converts a python string into numerical tensor IDs
        res = [self.stoi["<s>"]]
        for ch in text:
            res.append(self.stoi.get(ch, self.stoi["<unk>"]))
        res.append(self.stoi["</s>"])
        return res

    def decode(self, ids: list) -> str:
        # Converts index lists back into legible output strings
        return "".join([self.itos.get(i, "") for i in ids if i > 3])

class AssemblyDataset(Dataset):
    """Parses JSONL training collections into tokenized input/label tensors
    configured for causal language modeling objective.

    Loss is masked to the completion span only (see assembly_collate_fn) - the
    model shouldn't spend its limited capacity learning to predict the highly
    variable natural-language question text, only the JSON it needs to emit.
    """
    def __init__(self, file_path: str, tokenizer: AssemblyTokenizer, max_seq_len: int = 256):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                # Concatenate user input and target JSON completion into a singular context pass
                prefix = f"User: {data['prompt']}\nBot: "
                full_text = f"{prefix}{data['completion']}"
                tokens = self.tokenizer.encode(full_text)
                if len(tokens) <= self.max_seq_len:
                    # Number of characters before the completion begins. Combined with the
                    # leading <s> token, this marks the last input position that should NOT
                    # be supervised (see assembly_collate_fn) - everything from here onward is
                    # the actual completion the model needs to learn to produce.
                    prompt_char_len = len(prefix)
                    self.examples.append((torch.tensor(tokens, dtype=torch.long), prompt_char_len))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def assembly_collate_fn(batch, pad_value=0):
    """Dynamically groups and pads tensor slices inside an active mini-batch."""
    lengths = [len(x[0]) for x in batch]
    max_len = max(lengths)
    
    # Pre-populate matrix arrays with pad values
    inputs = torch.full((len(batch), max_len), pad_value, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long) # -100 tells PyTorch Loss to ignore calculations
    
    for i, (tokens, prompt_char_len) in enumerate(batch):
        inputs[i, :len(tokens)] = tokens
        # Shift targets down by 1 sequence position to train autoregressive prediction alignment
        labels[i, :len(tokens)-1] = tokens[1:]
        # Mask out the prompt span - only the completion (from "Bot: " onward) is supervised.
        labels[i, :prompt_char_len] = -100
        
    return inputs, labels
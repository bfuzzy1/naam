import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import os
import tempfile
import json
from safetensors.torch import save_model, load_model

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# this is not needed 
temp_dir = os.path.join(os.getcwd(), 'temp')
os.makedirs(temp_dir, exist_ok=True)
tempfile.tempdir = temp_dir
os.environ['TMPDIR'] = temp_dir

def get_ratio(teacher_logits, logits, mu=0.5):
    """Calculate adaptive importance ratios for knowledge distillation.
    
    Implementation based on Wu et al. (2024) "Rethinking Kullback-Leibler Divergence 
    in Knowledge Distillation for Large Language Models" (arXiv:2404.02657).
    
    Args:
        teacher_logits: Logits from teacher model [batch_size, seq_len, vocab_size]
        logits: Logits from student model [batch_size, seq_len, vocab_size]
        mu: Threshold for cumulative probability (default: 0.5)
        
    Returns:
        Tuple of (head_ratio, tail_ratio) representing importance weights for 
        head and tail probability distributions
    """
    teacher_logits = torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0).to(torch.float32)
    logits = torch.masked_fill(logits, torch.isinf(logits), 0).to(torch.float32)
    
    # calculate probability distributions
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32).detach()

    # sort teacher probabilities and get corresponding student probs
    re_teacher_probs, idx = teacher_probs.sort(dim=-1, descending=True)
    re_student_probs = student_probs.gather(dim=-1, index=idx)

    # calculate absolute errors between distributions
    errors = torch.abs(re_teacher_probs - re_student_probs)
    
    # create mask for head/tail separation based on cumulative probability
    cum_sum = torch.cumsum(re_teacher_probs, dim=-1)
    mask = cum_sum > mu
    mask[:,:,0] = False  # ensure first token is always included in head

    # calculate head and tail ratios
    s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)
    s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)

    return s1/(s1+s2), s2/(s1+s2)

def get_kl(teacher_logits, logits, inf_mask, mask, ratio=None):
    """Calculate weighted KL divergence between teacher and student distributions.
    
    Implementation based on Wu et al. (2024) "Rethinking Kullback-Leibler Divergence 
    in Knowledge Distillation for Large Language Models" (arXiv:2404.02657).
    
    Args:
        teacher_logits: Logits from teacher model
        logits: Logits from student model
        inf_mask: Mask for infinite values
        mask: Padding mask
        ratio: Optional importance weights for weighted KL divergence
        
    Returns:
        KL divergence loss value
    """
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_prod_probs = torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    teacher_x = torch.sum(teacher_prod_probs, dim=-1).view(-1)

    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)

    if ratio is None:
        distil_loss = torch.sum((teacher_x-x) * mask.view(-1)) / torch.sum(mask.view(-1))
    else:
        distil_loss = torch.sum((teacher_x-x) * ratio.view(-1) * mask.view(-1)) / torch.sum(mask.view(-1))
    return distil_loss

def AKL(teacher_logits, logits, label_mask):
    """Adaptive KL divergence loss combining head and tail distribution matching.
    
    Implementation based on Wu et al. (2024) "Rethinking Kullback-Leibler Divergence 
    in Knowledge Distillation for Large Language Models" (arXiv:2404.02657).
    
    This loss function adaptively balances the importance of matching high and low 
    probability tokens between teacher and student models, improving knowledge
    transfer across the entire probability distribution.
    
    Args:
        teacher_logits: Logits from teacher model
        logits: Logits from student model
        label_mask: Mask for padding tokens
        
    Returns:
        Combined adaptive KL divergence loss
    """
    inf_mask = torch.isinf(logits)
    mask = (label_mask != -100).int()
    
    h_ratio, l_ratio = get_ratio(teacher_logits, logits)
    distil_loss = get_kl(teacher_logits, logits, inf_mask, mask, h_ratio) + \
                  get_kl(logits, teacher_logits, inf_mask, mask, l_ratio)
    return distil_loss

class TxtDataset(Dataset):
    """Custom Dataset for text files"""
    def __init__(self, text, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        chunk_size = 256
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        # tokenize each chunk
        self.encodings = []
        for chunk in chunks:
            if not chunk.strip():  # skip empty chunks
                continue
                
            tokens = tokenizer(
                chunk,
                truncation=True,
                padding='max_length',
                max_length=self.seq_length,
                return_tensors="pt"
            ).input_ids[0]
            
            # keep sequences that have at least 10 tokens
            if len(tokens) >= 10:
                # pad
                if len(tokens) < self.seq_length:
                    padding = torch.full((self.seq_length - len(tokens),), tokenizer.pad_token_id)
                    tokens = torch.cat([tokens, padding])
                self.encodings.append(tokens[:self.seq_length])
        
        if not self.encodings:
            raise ValueError("No valid sequences found in the text. Check the input data.")
            
        self.encodings = torch.stack(self.encodings)
        print(f"Created dataset with {len(self.encodings)} sequences")
        
    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]

def load_data():
    """Load text from local data directory"""
    data_dir = Path("data")
    
    if not data_dir.exists():
        raise FileNotFoundError("data/ directory not found. Please create it and add your text files.")
    
    text_files = list(data_dir.glob("*.txt"))
    
    if not text_files:
        raise FileNotFoundError("No .txt files found in data/ directory")
    
    combined_text = ""
    for file_path in text_files:
        print(f"Reading file: {file_path}")
        text = file_path.read_text(encoding='utf-8')
        combined_text += text + "\n\n"
    
    if not combined_text:
        raise ValueError("Text files are empty")
    
    print(f"Loaded text with {len(combined_text)} characters")
    print(f"First 100 characters: {combined_text[:100]}")
    
    return combined_text

class ModelPruner:
    """Handles model pruning operations and activation statistics collection."""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.activation_sums = {}
        self.activation_counts = {}
        self.hooks = []

    def _save_activation(self, name):
        """Creates a hook function to save activation statistics."""
        def hook(module, input, output):
            if name not in self.activation_sums:
                self.activation_sums[name] = torch.zeros(output.shape[-1]).to(self.device)
                self.activation_counts[name] = 0
            
            dims_to_sum = tuple(range(len(output.shape)-1))
            self.activation_sums[name] += output.abs().sum(dim=dims_to_sum).detach()
            self.activation_counts[name] += torch.prod(torch.tensor(output.shape[:-1]))
        return hook

    def collect_activation_stats(self, model, dataloader):
        """Collects activation statistics for pruning."""
        # get intermediate size from the first MLP layer
        first_mlp = model.model.layers[0].mlp
        intermediate_size = first_mlp.gate_proj.out_features

        for idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                mlp.gate_proj = nn.Linear(mlp.gate_proj.in_features, intermediate_size, bias=False).to(self.device)
                mlp.up_proj = nn.Linear(mlp.up_proj.in_features, intermediate_size, bias=False).to(self.device)
                mlp.down_proj = nn.Linear(intermediate_size, mlp.down_proj.out_features, bias=False).to(self.device)
                
                for proj_name in ['gate_proj', 'up_proj']:
                    proj = getattr(mlp, proj_name)
                    name = f'layer_{idx}_{proj_name}'
                    hook = proj.register_forward_hook(self._save_activation(name))
                    self.hooks.append(hook)

        model.eval()
        model.to(self.device)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting Activations"):
                inputs = batch.to(self.device)
                model(inputs)

        for hook in self.hooks:
            hook.remove()

        avg_activations = {}
        for name in self.activation_sums:
            avg_activations[name] = self.activation_sums[name] / self.activation_counts[name]

        return avg_activations

    def prune_model(self, model, avg_activations, prune_percent=0.2):
        """Prunes model based on collected activation statistics."""
        # get intermediate size from the first MLP layer
        first_mlp = model.model.layers[0].mlp
        intermediate_size = first_mlp.gate_proj.out_features
        new_size = int(intermediate_size * (1 - prune_percent))

        for idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                gate_name = f'layer_{idx}_gate_proj'
                up_name = f'layer_{idx}_up_proj'
                
                # get activations for both projections
                gate_activations = avg_activations.get(gate_name)
                up_activations = avg_activations.get(up_name)
                
                if gate_activations is not None and up_activations is not None:
                    # combine activations
                    combined_activations = gate_activations + up_activations
                    
                    # get indices of neurons to keep
                    _, indices_to_keep = torch.topk(combined_activations, new_size, largest=True)
                    indices_to_keep = indices_to_keep.sort().values

                    # create new projections with consistent sizes
                    new_gate_proj = nn.Linear(mlp.gate_proj.in_features, new_size, bias=False).to(self.device)
                    new_up_proj = nn.Linear(mlp.up_proj.in_features, new_size, bias=False).to(self.device)
                    new_down_proj = nn.Linear(new_size, mlp.down_proj.out_features, bias=False).to(self.device)

                    # copy weights for kept neurons
                    new_gate_proj.weight.data = mlp.gate_proj.weight.data[indices_to_keep]
                    new_up_proj.weight.data = mlp.up_proj.weight.data[indices_to_keep]
                    new_down_proj.weight.data = mlp.down_proj.weight.data[:, indices_to_keep]

                    # update the layer
                    mlp.gate_proj = new_gate_proj
                    mlp.up_proj = new_up_proj
                    mlp.down_proj = new_down_proj

        return model

class KnowledgeDistiller:
    """Handles knowledge distillation process with adaptive KL divergence."""
    def __init__(self, teacher_model, student_model, device='cpu', save_path='models'):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        
        self.teacher_model.eval()
        self.student_model.train()
        self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=1e-5)
        
        self.best_loss = float('inf')
        self.frozen_layers_count = 0
        self.num_layers = len(student_model.model.layers)

    def _freeze_layers(self, start_idx, end_idx):
        """Freezes model layers from start_idx to end_idx."""
        newly_frozen = 0
        for idx in range(start_idx, end_idx):
            if idx >= self.num_layers:
                break
            layer = self.student_model.model.layers[idx]
            if any(p.requires_grad for p in layer.parameters()):
                for param in layer.parameters():
                    param.requires_grad = False
                newly_frozen += 1
        if newly_frozen > 0:
            self.frozen_layers_count += newly_frozen
            print(f"Froze {newly_frozen} layers ({start_idx} to {min(end_idx, self.num_layers-1)})")
        return newly_frozen
    
    def _visualize_layers(self):
        """Visualizes frozen and active layers status."""
        frozen = "❄️" * self.frozen_layers_count
        active = "🔥" * (self.num_layers - self.frozen_layers_count)
        print(f"\nLayer Status: [{frozen}{active}]")
        print(f"Frozen Layers: {self.frozen_layers_count}/{self.num_layers} ({self.frozen_layers_count/self.num_layers*100:.1f}%)")
    
    def train(self, dataloader, epochs=3, patience=2):
        """Performs knowledge distillation."""
        # load best model if it exists
        best_model_path = os.path.join(self.save_path, 'best_model.safetensors')
        if os.path.exists(best_model_path):
            try:
                state_dict = load_model(best_model_path)
                self.student_model.load_state_dict(state_dict)
                print(f"Loaded best model from {best_model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        best_loss = float('inf')
        plateau_counter = 0
        previous_loss = float('inf')
        
        # track frozen layers/schedules
        base_freeze_schedule = [(epoch, int((epoch / epochs) * self.num_layers * 0.8)) 
                               for epoch in range(epochs)]
        
        def print_epoch_banner(epoch):
            print(f"\n{'='*20} 🚀 EPOCH {epoch+1} 🚀 {'='*20}")
        
        def print_training_status(loss):
            if loss < 0.1:
                status = "🌟 Excellent!"
            elif loss < 0.3:
                status = "✨ Good Progress!"
            elif loss < 0.5:
                status = "💪 Keep Going!"
            else:
                status = "🔄 Still Learning..."
            print(f"\nDistillation Status: {status} (Loss: {loss:.4f})")
        
        for epoch in range(epochs):
            print_epoch_banner(epoch)
            total_loss = 0
            
            scheduled_frozen = base_freeze_schedule[epoch][1]
            if scheduled_frozen > self.frozen_layers_count:
                print(f"\n🌨️ Freezing new layers...")
                newly_frozen = self._freeze_layers(self.frozen_layers_count, scheduled_frozen)
                self._visualize_layers()
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"🏃 Distillation Progress")):
                inputs = batch.to(self.device)
                
                # attention mask (assuming no padding in sequence dimension)
                attention_mask = torch.ones_like(inputs, dtype=torch.bool)
                label_mask = attention_mask.clone()
                
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(inputs).logits
                student_outputs = self.student_model(inputs).logits
                
                loss = AKL(teacher_outputs, student_outputs, label_mask)
                
                total_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # check for plateau
                current_loss = loss.item()
                if current_loss >= previous_loss:
                    plateau_counter += 1
                    if plateau_counter >= patience:
                        # potentially freeze additional layers
                        unfrozen_layers = self.num_layers - self.frozen_layers_count
                        additional_freeze = max(1, int(unfrozen_layers * 0.1))
                        if self.frozen_layers_count + additional_freeze <= int(self.num_layers * 0.9):
                            newly_frozen = self._freeze_layers(self.frozen_layers_count, 
                                                          self.frozen_layers_count + additional_freeze)
                            self.frozen_layers_count += newly_frozen
                            self._visualize_layers()
                            # update base schedule
                            remaining_epochs = epochs - epoch
                            if remaining_epochs > 0:
                                for i in range(epoch + 1, epochs):
                                    base_freeze_schedule[i] = (i, max(base_freeze_schedule[i][1], 
                                                                self.frozen_layers_count))
                        plateau_counter = 0
                else:
                    plateau_counter = max(0, plateau_counter - 1)
                previous_loss = current_loss
                
                if batch_idx % 100 == 0:
                    print_training_status(loss.item())
            
            avg_loss = total_loss / len(dataloader)
            print(f"\n📊 Epoch Summary:")
            print(f"Average Loss: {avg_loss:.4f}")
            self._visualize_layers()
            
            if avg_loss < best_loss:
                print("🏆 New Best Model! Saving...")
                best_loss = avg_loss
                save_model(self.student_model, f"{self.save_path}/best_student_model.safetensors")
                
                metadata = {
                    'loss': str(best_loss),
                    'frozen_layers_count': str(self.frozen_layers_count)
                }
                with open(f"{self.save_path}/metadata.json", 'w') as f:
                    json.dump(metadata, f)
        
        print("\n🎉 Training Complete! 🎉")
        print(f"Final Model Status:")
        self._visualize_layers()
        print(f"Best Loss Achieved: {best_loss:.4f} 🌟")
        
        return self.student_model

def create_student_model(teacher_model, avg_activations, prune_percent=0.5, device='cpu'):
    """
    Creates and initializes a student model from the teacher model with pruned architecture.
    """
    # deep copy of the config
    config = teacher_model.config.to_dict()
    
    # fix pad_token_id
    if isinstance(config.get('pad_token_id', None), list):
        config['pad_token_id'] = config['pad_token_id'][0] if config['pad_token_id'] else None
    
    new_config = type(teacher_model.config)(**config)
    student_model = type(teacher_model)(new_config)
    
    student_model.load_state_dict(teacher_model.state_dict())
    
    pruner = ModelPruner(device=device)
    student_model = pruner.prune_model(student_model, avg_activations, prune_percent=prune_percent)
    student_model.to(device)
    return student_model

def main():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_safetensors=True
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = "<|begin_of_text|>"
    tokenizer.eos_token = "<|end_of_text|>"
    
    teacher_model.config.pad_token_id = tokenizer.pad_token_id
    teacher_model.config.bos_token_id = tokenizer.bos_token_id 
    teacher_model.config.eos_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    text = load_data()
    dataset = TxtDataset(text, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    pruner = ModelPruner(device=device)
    print("Collecting activation statistics...")
    avg_activations = pruner.collect_activation_stats(teacher_model, dataloader)
    
    print("Creating student model...")
    student_model = create_student_model(
        teacher_model,
        avg_activations,
        prune_percent=0.99,
        device=device
    )
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"Teacher model parameters: {teacher_params:,}")
    print(f"Student model parameters: {student_params:,}")
    print(f"Compression ratio: {teacher_params/student_params:.2f}x")

    # run distillation
    distiller = KnowledgeDistiller(
        teacher_model=teacher_model,
        student_model=student_model,
        device=device
    )
    
    student_model = distiller.train(
        dataloader=dataloader,
        epochs=3,
        patience=2
    )
    
    # test models
    test_prompt = "Hi "
    inputs = tokenizer(
        test_prompt,
        return_tensors="pt",
        add_special_tokens=True
    ).to(device)
    
    print("\nTeacher model output:")
    with torch.no_grad():
        teacher_output = teacher_model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            temperature=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    print(tokenizer.decode(teacher_output[0], skip_special_tokens=True))
    
    print("\nStudent model output:")
    with torch.no_grad():
        student_output = student_model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            temperature=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    print(tokenizer.decode(student_output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
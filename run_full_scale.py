import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from pyhessian import hessian
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import random
import copy

# ==========================================
# 1. CONFIGURATION (Overnight Mode)
# ==========================================
class Config:
    # System
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    SAVE_DIR = "bit_collapse_results"
    
    # Training (Rigorous)
    TRAIN_BATCH_SIZE = 64
    EPOCHS = 15              # Deep fine-tuning to find the true basin
    LR = 0.001               # Fine-tuning LR (Lower is better for stability)
    MOMENTUM = 0.9
    
    # SAM Settings
    SAM_RHO = 0.05           # Neighborhood size
    
    # Hessian Analysis (High Precision)
    HESSIAN_BATCH_SIZE = 4   # Keep small for VRAM safety
    NUM_HESSIAN_RUNS = 40    # Average over 20 batches to reduce variance/noise
    
    # Visualization
    PLOT_STEPS = 50          # High resolution curve
    PLOT_DISTANCE = 1.0      # Look +/- 1.0 units away

# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if Config.DEVICE == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True 

# ==========================================
# 2. CORE UTILS (Quant & SAM)
# ==========================================
class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8):
        qmin, qmax = -2**(num_bits-1), 2**(num_bits-1) - 1
        scale = x.abs().max() / qmax
        x_quant = (x / scale).round().clamp(qmin, qmax)
        return x_quant * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None 

class QuantLinear(nn.Linear):
    def forward(self, input):
        w_q = FakeQuantOp.apply(self.weight, 8)
        return F.linear(input, w_q, self.bias)

def make_model_quantized(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = QuantLinear(module.in_features, module.out_features, module.bias is not None)
            new_layer.weight = module.weight
            if module.bias is not None: new_layer.bias = module.bias
            setattr(model, name, new_layer)
        else:
            make_model_quantized(module)
    return model

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["e_w"] = p.grad * scale
                p.add_(self.state[p]["e_w"])
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        stack = [p.grad.norm(p=2).to(p.device) for group in self.param_groups for p in group["params"] if p.grad is not None]
        return torch.norm(torch.stack(stack), p=2)

# ==========================================
# 3. METRICS & VISUALIZATION (Manual)
# ==========================================
def get_loaders():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Full dataset for rigorous training
    loader = torch.utils.data.DataLoader(trainset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    return loader

def measure_hessian_robust(model, loader):
    """Measures Top Eigenvalue (Curvature) averaged over multiple batches."""
    print(f"   [Metric] Measuring Hessian (Avg over {Config.NUM_HESSIAN_RUNS} runs)...")
    model.eval()
    eigenvalues = []
    iterator = iter(loader)
    
    for i in range(Config.NUM_HESSIAN_RUNS):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader); inputs, targets = next(iterator)

        # Move small batch to GPU
        inputs = inputs[:Config.HESSIAN_BATCH_SIZE].to(Config.DEVICE)
        targets = targets[:Config.HESSIAN_BATCH_SIZE].to(Config.DEVICE)
        criterion = nn.CrossEntropyLoss()

        try:
            # PyHessian calculation
            hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=(Config.DEVICE == 'cuda'))
            top_eig, _ = hessian_comp.eigenvalues(top_n=1)
            eigenvalues.append(top_eig[0])
        except Exception as e:
            print(f"      [Warn] Batch {i} failed: {e}")
        
        # Cleanup
        del hessian_comp, inputs, targets, criterion
        torch.cuda.empty_cache()
        gc.collect()

    if not eigenvalues: return 0.0
    avg = np.mean(eigenvalues)
    std = np.std(eigenvalues)
    print(f"      -> Mean Lambda_max: {avg:.2f} ± {std:.2f}")
    return avg

def manual_loss_landscape(model, loader, title):
    """
    Manually walks in a random filter-normalized direction to plot 1D landscape.
    Avoids library dependencies/errors.
    """
    print(f"   [Visual] Tracing landscape for {title}...")
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Get a fixed batch for visualization
    inputs, targets = next(iter(loader))
    inputs = inputs.to(Config.DEVICE)
    targets = targets.to(Config.DEVICE)
    
    # 1. Create Random Direction (normalized by weight magnitude)
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        # Filter Normalization (Li et al., 2018)
        d = d * (p.data.norm() / (d.norm() + 1e-10))
        direction.append(d)
        
    # 2. Walk
    alphas = np.linspace(-Config.PLOT_DISTANCE, Config.PLOT_DISTANCE, Config.PLOT_STEPS)
    losses = []
    
    # Save original weights to restore later
    original_weights = [p.clone() for p in model.parameters()]
    
    for alpha in alphas:
        # Apply Perturbation: W_new = W_orig + alpha * direction
        for p, w_orig, d in zip(model.parameters(), original_weights, direction):
            p.data = w_orig + alpha * d
            
        with torch.no_grad():
            output = model(inputs)
            loss = criterion(output, targets).item()
            losses.append(loss)
            
    # Restore Model
    for p, w_orig in zip(model.parameters(), original_weights):
        p.data = w_orig
        
    return alphas, losses

def train_loop(model, loader, optimizer, use_sam=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        if use_sam:
            optimizer.first_step(zero_grad=True)
            criterion(model(inputs), targets).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()
            optimizer.zero_grad()

# ==========================================
# 4. MAIN EXPERIMENT
# ==========================================
def run():
    set_seed(Config.SEED)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    loader = get_loaders()
    print(f"--- OVERNIGHT BIT-COLLAPSE (Device: {Config.DEVICE}) ---")

    # -------------------------------------------------
    # PHASE 1: FP32 CONTROL (Train till convergence)
    # -------------------------------------------------
    print(f"\n[1] Training FP32 Control ({Config.EPOCHS} Epochs)...")
    model_fp32 = resnet18(weights=ResNet18_Weights.DEFAULT)
    model_fp32.fc = nn.Linear(512, 10)
    model_fp32 = model_fp32.to(Config.DEVICE)
    
    opt_fp32 = optim.SGD(model_fp32.parameters(), lr=Config.LR, momentum=Config.MOMENTUM)
    
    for ep in range(Config.EPOCHS):
        print(f"   FP32 Epoch {ep+1}/{Config.EPOCHS}")
        train_loop(model_fp32, loader, opt_fp32, use_sam=False)
        
    sharp_fp32 = measure_hessian_robust(model_fp32, loader)
    x_fp32, y_fp32 = manual_loss_landscape(model_fp32, loader, "FP32")
    
    # SAVE THE "GOOD" STATE
    converged_state = copy.deepcopy(model_fp32.state_dict())
    del model_fp32, opt_fp32
    torch.cuda.empty_cache()

    # -------------------------------------------------
    # PHASE 2: INT8 SGD (The Failure)
    # -------------------------------------------------
    print(f"\n[2] Training Int8 SGD ({Config.EPOCHS} Epochs)...")
    model_int8 = resnet18(weights=None)
    model_int8.fc = nn.Linear(512, 10)
    model_int8.load_state_dict(converged_state) # Start from good weights
    model_int8 = make_model_quantized(model_int8).to(Config.DEVICE)
    
    opt_int8 = optim.SGD(model_int8.parameters(), lr=Config.LR, momentum=Config.MOMENTUM)
    
    for ep in range(Config.EPOCHS):
        print(f"   Int8-SGD Epoch {ep+1}/{Config.EPOCHS}")
        train_loop(model_int8, loader, opt_int8, use_sam=False)
        
    sharp_int8 = measure_hessian_robust(model_int8, loader)
    x_int8, y_int8 = manual_loss_landscape(model_int8, loader, "Int8 SGD")
    del model_int8, opt_int8
    torch.cuda.empty_cache()

    # -------------------------------------------------
    # PHASE 3: INT8 SAM (The Fix)
    # -------------------------------------------------
    print(f"\n[3] Training Int8 SAM ({Config.EPOCHS} Epochs)...")
    model_sam = resnet18(weights=None)
    model_sam.fc = nn.Linear(512, 10)
    model_sam.load_state_dict(converged_state) # Start from good weights
    model_sam = make_model_quantized(model_sam).to(Config.DEVICE)
    
    base_opt = optim.SGD
    opt_sam = SAM(model_sam.parameters(), base_opt, rho=Config.SAM_RHO, lr=Config.LR, momentum=Config.MOMENTUM)
    
    for ep in range(Config.EPOCHS):
        print(f"   Int8-SAM Epoch {ep+1}/{Config.EPOCHS}")
        train_loop(model_sam, loader, opt_sam, use_sam=True)
        
    sharp_sam = measure_hessian_robust(model_sam, loader)
    x_sam, y_sam = manual_loss_landscape(model_sam, loader, "Int8 SAM")
    del model_sam
    torch.cuda.empty_cache()

    # -------------------------------------------------
    # RESULTS & PLOTTING
    # -------------------------------------------------
    print("\n--- FINAL REPORT ---")
    print(f"1. FP32 (Base): {sharp_fp32:.2f}")
    print(f"2. Int8 (SGD) : {sharp_int8:.2f}")
    print(f"3. Int8 (SAM) : {sharp_sam:.2f}")
    
    # Save Data
    with open(f"{Config.SAVE_DIR}/final_metrics.txt", "w") as f:
        f.write(f"FP32: {sharp_fp32}\nInt8-SGD: {sharp_int8}\nInt8-SAM: {sharp_sam}\n")

    # Generate Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_fp32, y_fp32, label='FP32 (Baseline)', color='blue', linewidth=2, linestyle='--')
    plt.plot(x_int8, y_int8, label='Int8 SGD (Bit-Collapse)', color='red', linewidth=2)
    plt.plot(x_sam, y_sam, label='Int8 SAM (Restored)', color='green', linewidth=2)
    
    plt.title(f"Bit-Collapse: Geometric Instability of Quantization", fontsize=14)
    plt.xlabel("Step Size (Filter Normalized)", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # IMPORTANT: Zoom in on the basin bottom
    plt.ylim(0, 5.0) 
    
    plt.savefig(f"{Config.SAVE_DIR}/bit_collapse_overnight.png", dpi=300)
    print(f"✅ Success. Plot saved to {Config.SAVE_DIR}/bit_collapse_overnight.png")

if __name__ == "__main__":
    run()

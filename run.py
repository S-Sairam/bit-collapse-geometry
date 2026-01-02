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
import time

# ==========================================
# 1. CONFIGURATION (Control Panel)
# ==========================================
class Config:
    # System
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Memory Management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Experiment Switches
    RUN_SAM = True         # Set to True to generate the Green Line (The Fix)
    
    # Training
    TRAIN_BATCH_SIZE = 64
    EPOCHS = 5             # Enough to trigger collapse
    LR = 0.01
    MOMENTUM = 0.9
    
    # Hessian Analysis
    HESSIAN_BATCH_SIZE = 4 # Tiny batch to fit in 4GB VRAM
    NUM_HESSIAN_RUNS = 10  # Monte Carlo averaging for statistical rigor
    
    # Visualization
    PLOT_STEPS = 20
    PLOT_DISTANCE = 0.5    # Zoom level (Keep small to see the bottom)
    SAVE_DIR = "bit_collapse_results"

# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if Config.DEVICE == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True # Optimized for ResNet

# ==========================================
# 2. QUANTIZATION ENGINE (Bit-Collapse Logic)
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
        return grad_output, None # Straight-Through Estimator (STE)

class QuantLinear(nn.Linear):
    def forward(self, input):
        w_q = FakeQuantOp.apply(self.weight, 8)
        return F.linear(input, w_q, self.bias)

def make_model_quantized(model):
    """Recursively replaces Linear layers with Quantized versions."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = QuantLinear(module.in_features, module.out_features, module.bias is not None)
            new_layer.weight = module.weight
            if module.bias is not None: new_layer.bias = module.bias
            setattr(model, name, new_layer)
        else:
            make_model_quantized(module)
    return model

# ==========================================
# 3. OPTIMIZERS (Standard & Geometric)
# ==========================================
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
# 4. MEASUREMENT & VISUALIZATION
# ==========================================
def get_loaders():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Using a subset for faster demonstration? No, full dataset for rigorous results.
    loader = torch.utils.data.DataLoader(trainset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    return loader

def measure_hessian_robust(model, loader):
    """Monte Carlo Hessian Estimation: Avgs top eigenvalue over multiple batches"""
    print(f"   [Metric] Measuring Hessian Spectrum (Avg over {Config.NUM_HESSIAN_RUNS} batches)...")
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
            hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=(Config.DEVICE == 'cuda'))
            top_eig, _ = hessian_comp.eigenvalues(top_n=1)
            eigenvalues.append(top_eig[0])
        except Exception as e:
            print(f"      [Warning] Batch {i} calc failed: {e}")
        
        # Aggressive Cleanup
        del hessian_comp, inputs, targets, criterion
        torch.cuda.empty_cache()
        gc.collect()

    if not eigenvalues: return 0.0
    avg = np.mean(eigenvalues)
    std = np.std(eigenvalues)
    print(f"      -> Mean Lambda_max: {avg:.2f} ± {std:.2f}")
    return avg

def get_loss_curve(model, loader, title):
    """Generates 1D loss landscape curve"""
    print(f"   [Visual] Tracing landscape for {title}...")
    model.eval()
    criterion = nn.CrossEntropyLoss()
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
    
    # Generate random normalized direction
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        d = d * (p.norm() / (d.norm() + 1e-10)) # Filter Normalization
        direction.append(d)
        
    alphas = np.linspace(-Config.PLOT_DISTANCE, Config.PLOT_DISTANCE, Config.PLOT_STEPS)
    losses = []
    
    # Save original weights
    orig_weights = [p.clone() for p in model.parameters()]
    
    for alpha in alphas:
        # Perturb
        for p, d, w0 in zip(model.parameters(), direction, orig_weights):
            p.data = w0 + alpha * d
        with torch.no_grad():
            losses.append(criterion(model(inputs), targets).item())
            
    # Restore
    for p, w0 in zip(model.parameters(), orig_weights):
        p.data = w0
        
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
# 5. MAIN EXPERIMENT
# ==========================================
def run():
    set_seed(Config.SEED)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    loader = get_loaders()
    print(f"--- EXPERIMENT STARTED (SAM={Config.RUN_SAM}) ---")

    # --- 1. FP32 CONTROL ---
    print("\n[1] Training FP32 Baseline...")
    model_fp32 = resnet18(weights=ResNet18_Weights.DEFAULT)
    model_fp32.fc = nn.Linear(512, 10)
    model_fp32 = model_fp32.to(Config.DEVICE)
    optimizer = optim.SGD(model_fp32.parameters(), lr=Config.LR, momentum=Config.MOMENTUM)
    
    # Train briefly to settle in basin
    for ep in range(Config.EPOCHS):
        train_loop(model_fp32, loader, optimizer)
        
    sharp_fp32 = measure_hessian_robust(model_fp32, loader)
    x_fp32, y_fp32 = get_loss_curve(model_fp32, loader, "FP32")
    
    # Save state to start Int8 from here
    start_state = copy.deepcopy(model_fp32.state_dict())
    del model_fp32, optimizer
    torch.cuda.empty_cache()

    # --- 2. INT8 SGD (FAILURE) ---
    print("\n[2] Training Int8 SGD (Bit-Collapse)...")
    model_int8 = resnet18(weights=None)
    model_int8.fc = nn.Linear(512, 10)
    model_int8.load_state_dict(start_state)
    model_int8 = make_model_quantized(model_int8).to(Config.DEVICE)
    
    optimizer = optim.SGD(model_int8.parameters(), lr=Config.LR, momentum=Config.MOMENTUM)
    for ep in range(Config.EPOCHS):
        train_loop(model_int8, loader, optimizer)
        
    sharp_int8 = measure_hessian_robust(model_int8, loader)
    x_int8, y_int8 = get_loss_curve(model_int8, loader, "Int8 SGD")
    del model_int8, optimizer
    torch.cuda.empty_cache()

    # --- 3. INT8 SAM (FIX - OPTIONAL) ---
    x_sam, y_sam, sharp_sam = None, None, 0.0
    if Config.RUN_SAM:
        print("\n[3] Training Int8 SAM (Restoration)...")
        model_sam = resnet18(weights=None)
        model_sam.fc = nn.Linear(512, 10)
        model_sam.load_state_dict(start_state)
        model_sam = make_model_quantized(model_sam).to(Config.DEVICE)
        
        base = optim.SGD
        optimizer = SAM(model_sam.parameters(), base, lr=Config.LR, momentum=Config.MOMENTUM)
        for ep in range(Config.EPOCHS):
            train_loop(model_sam, loader, optimizer, use_sam=True)
            
        sharp_sam = measure_hessian_robust(model_sam, loader)
        x_sam, y_sam = get_loss_curve(model_sam, loader, "Int8 SAM")
        del model_sam
        torch.cuda.empty_cache()

    # --- REPORTING ---
    print("\n--- RESULTS ---")
    print(f"FP32 Sharpness: {sharp_fp32:.2f}")
    print(f"Int8 Sharpness: {sharp_int8:.2f}")
    if Config.RUN_SAM: print(f"SAM Sharpness : {sharp_sam:.2f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_fp32, y_fp32, label='FP32 (Baseline)', color='blue', linewidth=2, linestyle='--')
    plt.plot(x_int8, y_int8, label='Int8 SGD (Bit-Collapse)', color='red', linewidth=2)
    if Config.RUN_SAM:
        plt.plot(x_sam, y_sam, label='Int8 SAM (Restored)', color='green', linewidth=2)
    
    plt.title(f"Geometry of Bit-Collapse (Ratio: {sharp_int8/sharp_fp32:.1f}x)")
    plt.xlabel("Step Size")
    plt.ylabel("Loss")
    plt.ylim(0, 5.0) # Zoom in
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{Config.SAVE_DIR}/bit_collapse_plot.png", dpi=300)
    print(f"✅ Plot saved to {Config.SAVE_DIR}")

if __name__ == "__main__":
    run()

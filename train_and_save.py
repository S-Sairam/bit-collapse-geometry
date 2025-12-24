import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from pyhessian import hessian
from quant import FakeQuantOp, QuantLinear, make_model_quantized
import os

# --- CONFIG ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = 'cuda'
BATCH_SIZE = 32
HESSIAN_BATCH = 4
EPOCHS = 2  # Keep it short for MVP

# --- SAM OPTIMIZER CLASS (Paste here or import) ---
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

def train_loop(model, loader, optimizer, sam=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        if sam:
            optimizer.first_step(zero_grad=True)
            criterion(model(x), y).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()
            optimizer.zero_grad()

def measure(model, loader):
    print("   [Metric] Measuring Hessian Spectrum...")
    model.eval()
    model = model.to(DEVICE) # Ensure model is on GPU
    
    # 1. Get Data
    try:
        x, y = next(iter(loader))
    except StopIteration:
        # Handle case if loader is empty
        loader = get_loaders()
        x, y = next(iter(loader))
        
    # 2. Force Move to Device (The Fix)
    x = x[:HESSIAN_BATCH].to(DEVICE)
    y = y[:HESSIAN_BATCH].to(DEVICE)
    
    # Debug print to confirm location
    # print(f"DEBUG: Input device: {x.device}, Model device: {next(model.parameters()).device}")

    # 3. Compute
    criterion = nn.CrossEntropyLoss()
    hessian_comp = hessian(model, criterion, data=(x, y), cuda=True)
    top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=1)
    
    return top_eigenvalues[0]

def main():
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    # ---------------------------------------------------------
    # 1. Baseline FP32
    # ---------------------------------------------------------
    print("Training FP32...")
    model_fp32 = resnet18(weights=ResNet18_Weights.DEFAULT)
    model_fp32.fc = nn.Linear(512, 10) # Replace head first
    model_fp32 = model_fp32.to(DEVICE) # Move to GPU LAST
    
    optimizer = optim.SGD(model_fp32.parameters(), lr=0.01)
    
    s_fp32 = measure(model_fp32, loader)
    print(f"FP32 Sharpness: {s_fp32}")
    torch.save(model_fp32.state_dict(), "fp32.pt")
    
    # Cleanup to save VRAM
    del model_fp32, optimizer
    torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # 2. Int8 SGD
    # ---------------------------------------------------------
    print("\nTraining Int8 SGD...")
    # Step A: Load Pretrained
    model_int8 = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Step B: Replace Head (So it matches CIFAR classes)
    model_int8.fc = nn.Linear(512, 10)
    # Step C: Quantize (Wraps the new head too!)
    model_int8 = make_model_quantized(model_int8)
    # Step D: Move to Device LAST
    model_int8 = model_int8.to(DEVICE)
    
    optimizer = optim.SGD(model_int8.parameters(), lr=0.01)
    for _ in range(EPOCHS): train_loop(model_int8, loader, optimizer)
    
    s_int8 = measure(model_int8, loader)
    print(f"Int8 SGD Sharpness: {s_int8}")
    torch.save(model_int8.state_dict(), "int8_sgd.pt")

    del model_int8, optimizer
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 3. Int8 SAM
    # ---------------------------------------------------------
    print("\nTraining Int8 SAM...")
    # Same order of operations
    model_sam = resnet18(weights=ResNet18_Weights.DEFAULT)
    model_sam.fc = nn.Linear(512, 10)
    model_sam = make_model_quantized(model_sam)
    model_sam = model_sam.to(DEVICE) # Move to GPU LAST
    
    base = optim.SGD
    optimizer = SAM(model_sam.parameters(), base, lr=0.01)
    for _ in range(EPOCHS): train_loop(model_sam, loader, optimizer, sam=True)
    
    s_sam = measure(model_sam, loader)
    print(f"Int8 SAM Sharpness: {s_sam}")
    torch.save(model_sam.state_dict(), "int8_sam.pt")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import loss_landscapes
import loss_landscapes.metrics
import matplotlib.pyplot as plt
from quant import make_model_quantized

DEVICE = 'cpu' # CPU for plotting safety
STEPS = 30
DISTANCE = 0.5 # Zoom level

def get_curve(path, quantized, name, x, y):
    print(f"Plotting {name}...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    if quantized: model = make_model_quantized(model)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    
    criterion = nn.CrossEntropyLoss()
    metric = loss_landscapes.metrics.Loss(criterion, x, y)
    return loss_landscapes.random_line(model, metric, distance=DISTANCE, steps=STEPS, normalization='filter', deepcopy_model=True)

def main():
    # Load 1 batch of data
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    x, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)))

    y_fp32 = get_curve("fp32.pt", False, "FP32", x, y)
    y_int8 = get_curve("int8_sgd.pt", True, "Int8 SGD", x, y)
    y_sam = get_curve("int8_sam.pt", True, "Int8 SAM", x, y)

    plt.figure(figsize=(10, 6))
    x_axis = range(len(y_fp32))
    plt.plot(x_axis, y_fp32, label='FP32 (Baseline)', color='blue', linestyle='--')
    plt.plot(x_axis, y_int8, label='Int8 SGD (Collapse)', color='red')
    plt.plot(x_axis, y_sam, label='Int8 SAM (Fix)', color='green')
    
    # CRITICAL: CROP THE VIEW
    plt.ylim(0, 5.0) 
    
    plt.legend()
    plt.title("Bit-Collapse: Loss Landscape Geometry")
    plt.savefig("bit_collapse_final.png")
    print("✅ Done.")

if __name__ == "__main__":
    main()

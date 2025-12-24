import torch
import torch.nn as nn
import torch.nn.functional as F

# The Straight-Through Estimator (STE)
class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8):
        qmin, qmax = -128, 127
        scale = x.abs().max() / 127
        x_quant = (x / scale).round().clamp(qmin, qmax)
        return x_quant * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

# The Layer Replacement
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

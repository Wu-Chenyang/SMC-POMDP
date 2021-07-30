import torch

def global_grad_norm(params):
    grad_norm = 0.0
    for param in params:
        if param.grad == None:
            continue
        grad_norm = max(grad_norm, torch.max(torch.abs(param.grad.data)).item())
    return grad_norm
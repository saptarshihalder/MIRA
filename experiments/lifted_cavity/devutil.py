"""Device helpers: data stays on the CPU and each batch moves to the model's device just before the forward pass."""
import torch


def pick(device='auto'):
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def to_dev(x, dev):
    if torch.is_tensor(x):
        return x.to(dev, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_dev(v, dev) for k, v in x.items()}
    return x


def model_device(model):
    return next(model.parameters()).device

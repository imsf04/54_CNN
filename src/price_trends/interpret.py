from __future__ import annotations
import torch

def grad_cam(model, image, class_index=None):
    """Return a normalized Grad-CAM heatmap for one image."""
    activations = []; gradients = []
    layer = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)][-1]
    h1 = layer.register_forward_hook(lambda _, __, out: activations.append(out))
    h2 = layer.register_full_backward_hook(lambda _, grad_in, grad_out: gradients.append(grad_out[0]))
    model.zero_grad(set_to_none=True); logits = model(image); index = int(logits.argmax(1)[0]) if class_index is None else class_index
    logits[:, index].sum().backward(); h1.remove(); h2.remove()
    weights = gradients[0].mean(dim=(2, 3), keepdim=True); cam = (weights * activations[0]).sum(1).relu()[0]
    cam = cam - cam.min(); return cam / cam.max().clamp_min(1e-8)

def ablation_configs():
    return [{"name": "baseline", "channels": (16, 32, 64), "blocks": 3}, {"name": "shallow", "channels": (16, 32), "blocks": 2}, {"name": "wide", "channels": (32, 64, 128), "blocks": 3}]

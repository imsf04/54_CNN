from __future__ import annotations

import torch


def grad_cam(model: torch.nn.Module, image: torch.Tensor, class_index: int | None = None) -> torch.Tensor:
    """Return a normalized Grad-CAM map for the final convolution layer."""

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)][-1]
    forward_hook = layer.register_forward_hook(lambda _module, _inputs, output: activations.append(output))
    backward_hook = layer.register_full_backward_hook(
        lambda _module, _grad_input, grad_output: gradients.append(grad_output[0])
    )
    model.zero_grad(set_to_none=True)
    logits = model(image)
    index = int(logits.argmax(dim=1)[0]) if class_index is None else class_index
    logits[:, index].sum().backward()
    forward_hook.remove()
    backward_hook.remove()
    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations[0]).sum(dim=1).relu()[0]
    cam = cam - cam.min()
    return cam / cam.max().clamp_min(1e-8)

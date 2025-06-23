import torch
import torch.nn.functional as F
from torch import nn
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = dict(model.named_modules())[target_layer_name]
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward()

        pooled_grad = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        weighted_activation = pooled_grad * self.activations
        cam = torch.sum(weighted_activation, dim=1).squeeze()  # [T, N]
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx
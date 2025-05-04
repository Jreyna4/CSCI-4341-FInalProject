import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.sigmoid().item() > 0.5
        loss = output[:, class_idx] if output.ndim == 2 else output
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def get_last_conv_layer(model):
    # For torchvision DenseNet and EfficientNet
    if hasattr(model, 'features'):
        # DenseNet
        for module in reversed(list(model.features.children())):
            if isinstance(module, torch.nn.Conv2d):
                return module
            if hasattr(module, 'conv'):  # For _DenseLayer
                return module.conv
    elif hasattr(model, 'features') and hasattr(model.features, 'children'):
        # EfficientNet
        for module in reversed(list(model.features.children())):
            if isinstance(module, torch.nn.Conv2d):
                return module
    raise ValueError('Could not find last conv layer.')

def generate_and_save_gradcam(model, image_path, save_path, device):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    last_conv = get_last_conv_layer(model)
    gradcam = GradCAM(model, last_conv)
    cam = gradcam(input_tensor)
    gradcam.remove_hooks()
    # Overlay heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(np.array(img))
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title('Grad-CAM')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close() 
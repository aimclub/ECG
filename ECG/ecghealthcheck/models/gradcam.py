import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model):
        super(GradCAM, self).__init__()
        self.model = model
        self.model.eval()
        self.features = []
        self.grad = []
        self.hooks = []

    def save_features(self, module, input, output):
        self.features.append(output.detach())

    def save_grad(self, module, grad_input, grad_output):
        self.grad.append(grad_output)

    def register_hooks(self, module):
        self.hooks.append(module.register_forward_hook(self.save_features))
        self.hooks.append(module.register_full_backward_hook(self.save_grad))

    def compute_grads(self, x1, x2):
        emb1 = self.model.forward(x1)
        emb2 = self.model.forward(x2)

        dist = torch.sqrt(torch.sum(torch.square(emb1 - emb2)))
        dist.backward()

        cam_results = []

        for i in range(len(self.grad)):
            pooled_grads = F.adaptive_avg_pool1d(self.grad[i][0], 1)
            weights = torch.relu(torch.sum(pooled_grads, dim=2))
            weighted_features = torch.mul(self.features[i], weights.unsqueeze(2))
            cam = torch.sum(weighted_features, dim=1)
            if torch.max(cam) > 0:
                cam = F.relu(cam)
            cam = cam / torch.max(cam)
            cam_results.append(cam)

        return cam_results

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

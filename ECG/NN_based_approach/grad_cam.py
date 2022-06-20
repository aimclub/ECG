import torch
import numpy as np
import cv2
from ECG.NN_based_approach.losses import custom_loss
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection


class ActGrad:
    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu())

    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model.forward(x)


class GradCam:
    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        self.model = model.eval()
        self.loss = custom_loss(None, 1.)
        self.target_layer = target_layer
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = self.model.cuda()
        self.reshape_transform = reshape_transform
        self.act_and_grad = ActGrad(self.model, target_layer, reshape_transform)

    def get_cam_weights(self, input_tensor,
                        target_category,
                        activations,
                        grads):
        grads_p2 = grads ** 2
        grads_p3 = grads_p2 * grads
        activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_p2 / (2 * grads_p2 + activations[:, :, None, None] * grads_p3 + eps)
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        return weights

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self,
                      input_tensor,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_tensor, target=None, eigen_smooth=False):

        if self.use_cuda:
            input_tensor = input_tensor.cuda()

        output = self.act_and_grad(input_tensor)

        if target is None:
            target = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert (len(target) == input_tensor.size(0))

        self.model.zero_grad()
        target1 = torch.Tensor(target)
        if self.use_cuda:
            target1 = target1.cuda()
        loss = self.loss(output, target1)
        loss.backward(retain_graph=True)

        activations = self.act_and_grad.activations[-1].cpu().data.numpy()
        grads = self.act_and_grad.gradients[-1].cpu().data.numpy()

        cam = self.get_cam_image(input_tensor, target,
                                 activations, grads, eigen_smooth)

        cam = np.maximum(cam, 0)

        result = []
        for img in cam:
            img = cv2.resize(img, input_tensor.shape[-2:][::-1])
            img = img - np.min(img)
            img = img / (np.max(img) - np.min(img) + 1e-8)
            result.append(img)

        result = np.float32(result)
        self.model.zero_grad()
        return result, output

    def __call__(self,
                 input_tensor,
                 target_category=None,
                 eigen_smooth=False):

        return self.forward(input_tensor, target_category, eigen_smooth)

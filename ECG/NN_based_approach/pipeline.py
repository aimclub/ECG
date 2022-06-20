import numpy as np
import torch
import os
import torch.nn as nn
from typing import Tuple, Any
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from ECG.data_classes import ElevatedST, NNResult
from ECG.NN_based_approach.utils import signal_rescale
from ECG.NN_based_approach.model_factory import create_model
from ECG.NN_based_approach.NN_Enums import NetworkType, ModelType
from ECG.NN_based_approach.grad_cam import GradCam
import matplotlib
matplotlib.use('Agg')


def _preprocess(signal: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)


def _signal_to_img(signal: np.ndarray) -> np.ndarray:
    img = signal - signal.min()
    img = img / img.max()
    return np.stack([img] * 3, axis=-1)


def predict(signal, net):
    input_tensor = _preprocess(signal)

    with torch.no_grad():
        prob = net.forward(input_tensor).cpu().detach().item()

    return prob


def explain(signal, cam):
    input_tensor = _preprocess(signal)

    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = Image.fromarray(show_cam_on_image(
        _signal_to_img(signal), grayscale_cam, use_rgb=False))
    return visualization


def is_BER(signal: np.ndarray, threshold: float,
           gradcam_enabled: bool, layered_images: bool) -> NNResult:
    signal = signal_rescale(signal, up_slice=5000)
    net = create_model(net_type=NetworkType.Conv, model_type=ModelType.BER)

    images = []
    if gradcam_enabled:
        images = _gradcam(net, signal, threshold=threshold, layered_images=layered_images,
                          save_path='./ECG/NN_based_approach/imgs', tag='BER')
    prob = predict(signal, net)

    return NNResult(prob, images)


def is_MI(signal: np.ndarray, threshold: float,
          gradcam_enabled: bool, layered_images: bool) -> NNResult:
    signal = signal_rescale(signal, up_slice=5000)
    net = create_model(net_type=NetworkType.Conv, model_type=ModelType.MI)

    images = []
    if gradcam_enabled:
        images = _gradcam(net, signal, threshold=threshold, layered_images=layered_images,
                          save_path='./ECG/NN_based_approach/imgs', tag='MI')
    prob = predict(signal, net)

    return NNResult(prob, images)


def check_STE(signal: np.ndarray) -> NNResult:
    signal = signal_rescale(signal, up_slice=4000)
    net = create_model(net_type=NetworkType.Conv1,
                       model_type=ModelType.STE, input_shape=(12, 4000))
    cam = GradCAM(model=net, target_layers=[net.conv4], use_cuda=False)
    prob = predict(signal, net)
    visualization = explain(signal, cam)

    return NNResult(prob, [visualization])


def _gradcam(net: nn.Module, signal: np.array, threshold: float, layered_images: bool,
             save_path: str = None, tag: str = None) -> [Image]:

    images = []
    cams = []
    layers = [net.conv1, net.conv2, net.conv3,
              net.conv4] if layered_images else [net.conv2]

    for layer in layers:
        cams.append(GradCam(model=net, target_layer=layer,
                    use_cuda=torch.cuda.is_available()))

    for layer in range(len(cams)):
        cam = cams[layer]

        img = _preprocess(signal)[0]
        cls = torch.tensor([1.])
        (res, pred) = cam(input_tensor=img.unsqueeze(0), target_category=cls)

        predicted = int(pred.item() >= threshold)
        name = '{}_layer{}_Predicted{}'.format(tag, layer, predicted) if layered_images\
            else '{}_Predicted{}'.format(tag, predicted)

        res = np.array(res.squeeze(0))
        res_range = np.max(res, axis=1, keepdims=True) - \
            np.min(res, axis=1, keepdims=True)
        res = (res - np.min(res, axis=1, keepdims=True))
        res = (res / (res_range + 0.0000001)) * 0.9

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(res.shape[0], figsize=(15, 12))
        for i in range(res.shape[0]):
            ax[i].vlines(list(range(3000)),
                         ymin=np.min(signal[i]), ymax=np.max(signal[i]),
                         alpha=np.clip(res.T[0:3000, i], 0.1, 0.9) ** 2, colors='grey')
            ax[i].plot(signal.T[0:3000, i])
            ax[i].grid()

        plt.suptitle('Tag = {} | Layer = {} | Predicted = {}'
                     .format(tag, layer if layered_images else '-', round(pred.item())))
        net.zero_grad()

        plt.gcf().canvas.get_renderer()

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig('{}/{}.png'.format(save_path, name))

        im = Image.frombytes('RGB', fig.canvas.get_width_height(),
                             fig.canvas.tostring_rgb())
        images.append(im)

        plt.close('all')
        plt.clf()

        del fig, ax
        del img, res, pred
        torch.cuda.empty_cache()

        del cam
        with torch.no_grad():
            torch.cuda.empty_cache()

    return images

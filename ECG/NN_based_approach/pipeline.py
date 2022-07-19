import numpy as np
import torch
from typing import Tuple
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from ECG.NN_based_approach.utils import signal_rescale
from ECG.data_classes import ElevatedST
from ECG.NN_based_approach.model_factory import create_model
from ECG.NN_based_approach.NN_Enums import NetworkType, ModelType


def _preprocess(signal: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)


def _signal_to_img(signal: np.ndarray) -> np.ndarray:
    img = signal - signal.min()
    img = img / img.max()
    return np.stack([img] * 3, axis=-1)


def predict_and_explain(signal, net, cam):
    input_tensor = _preprocess(signal)

    # predict
    with torch.no_grad():
        prob = net.forward(input_tensor).cpu().detach().item()

    # gradcam
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = Image.fromarray(show_cam_on_image(
        _signal_to_img(signal), grayscale_cam, use_rgb=True))

    return prob, visualization


def is_BER(signal: np.ndarray) -> Tuple[bool, bool]:
    # returns bool, probability, GradCAM Image
    signal = signal_rescale(signal, up_slice=5000)
    net = create_model(net_type=NetworkType.Conv, model_type=ModelType.BER)
    cam = GradCAM(model=net, target_layers=[net.conv4], use_cuda=False)
    prob, visualization = predict_and_explain(signal, net, cam)
    return (prob > 0.6, prob, visualization)


def is_MI(signal: np.ndarray) -> Tuple[bool, str]:
    # returns bool, probability, GradCAM Image
    signal = signal_rescale(signal, up_slice=5000)
    net = create_model(net_type=NetworkType.Conv, model_type=ModelType.MI)
    cam = GradCAM(model=net, target_layers=[net.conv4], use_cuda=False)
    prob, visualization = predict_and_explain(signal, net, cam)
    return (prob > 0.6, prob, visualization)


def check_STE(signal: np.ndarray) -> Tuple[ElevatedST, str]:
    # returns ElevatedST, probability, GradCAM Image
    signal = signal_rescale(signal, up_slice=4000)
    net = create_model(net_type=NetworkType.Conv1,
                       model_type=ModelType.STE, input_shape=(12, 4000))
    cam = GradCAM(model=net, target_layers=[net.conv4], use_cuda=False)
    prob, visualization = predict_and_explain(signal, net, cam)
    ste = ElevatedST.Present if prob > 0.6 else ElevatedST.Abscent
    return (ste, prob, visualization)

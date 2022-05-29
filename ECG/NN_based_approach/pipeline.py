from ECG.NN_based_approach.Networks.ConvNet import ConvNet
from ECG.NN_based_approach.NNType import NNType
import numpy as np
import torch
import torch.nn as nn


def create_model(net_type: NNType) -> nn:
    device = torch.device('cpu')
    model_dir = './ECG/NN_based_approach/Models/'

    if net_type == NNType.Conv:
        net = ConvNet(n_classes=1)
    else:
        assert False, "Key 'net_type' must be one of NNType enum"

    model_path = model_dir + net_type.value + '_model.pt'
    print("Load model at", model_path)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net


def process_recording(signal: np.ndarray, net: ConvNet) -> np.double:
    x_val = torch.from_numpy(signal).float().reshape(1, 1, 12, 5000)
    with torch.no_grad():
        pred = net.forward(x_val)
    return round(pred.cpu().detach().item(), 4)

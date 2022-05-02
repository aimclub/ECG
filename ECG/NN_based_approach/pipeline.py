from ECG.NN_based_approach.Networks.ConvNet1 import ConvNet1
from ECG.NN_based_approach.Networks.ConvNet import ConvNet
import numpy as np
import torch

def process_recording(signal: np.ndarray, model_name: str) -> np.double:
    device = torch.device('cpu')

    net = ConvNet(n_classes=1) # TODO: add model_name and net class connection
    net.load_state_dict(torch.load(model_name, map_location=device))
    net.eval()

    x_val = torch.from_numpy(signal).float().reshape(1,1,12,5000)
    pred = net.forward(x_val.to(torch.device('cpu')))
    return pred.cpu().detach().item()


from ECG.NN_based_approach.Networks.ConvNet import ConvNet, ConvNet1
from ECG.NN_based_approach.NN_Enums import NetworkType, ModelType
import torch
import torch.nn as nn


def create_model(net_type: NetworkType, model_type: ModelType, input_shape=(12, 5000)) -> nn:
    device = torch.device('cpu')
    model_dir = './ECG/NN_based_approach/Models/'

    if net_type == NetworkType.Conv:
        net = ConvNet(n_classes=1, input_shape=input_shape)
    elif net_type == NetworkType.Conv1:
        net = ConvNet1(n_classes=1, input_shape=input_shape)
    else:
        assert False, "Key 'net_type' must be one of NNType enum"

    model_path = model_dir + net_type.value + '_' + model_type.value + '_model.pt'
    print("Load model at", model_path)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net

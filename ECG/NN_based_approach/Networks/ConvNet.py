import torch.nn as nn
from ECG.NN_based_approach.utils import shape_change_conv, make_standard_double_layer


class ConvNet(nn.Module):
    def __init__(self, input_shape=(12, 5000), n_classes=12):
        super(ConvNet, self).__init__()
        '''
        This is basic convolutional neural network that is usually applied to computer vision tasks,
        but can be applied also in the sequence data analysis. In this neural network is also used
        residual connection to improve results.
        Parameters:
        -input shape - tuple (embedding size, sequence length). For ECG embedding size is number of predictors
        -n_classes - the number of classes that neural network fit to predict.
        '''
        # self.pos_enc = PositionalEncoder(input_shape[0], input_shape[1])

        out_shape = input_shape

        self.out_shape1 = out_shape

        dropout_coef = 0
        conv_ch = 20

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_ch, kernel_size=(7, 101),
                      padding=(3, 30), stride=(2, 4)),
            nn.BatchNorm2d(conv_ch),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=5, padding=2, stride=1)
        )
        out_shape = shape_change_conv(out_shape, kernel_size=(7, 101),
                                      padding=(3, 30), stride=(2, 4))
        self.conv2 = make_standard_double_layer(in_ch=conv_ch, out_ch=conv_ch, kernel_size=(5, 51),
                                                padding=(2, 25), stride=(1, 2), dropout_coef=dropout_coef)
        out_shape = shape_change_conv(out_shape, kernel_size=(5, 51),
                                      padding=(2, 25), stride=(1, 2))
        self.conv3 = make_standard_double_layer(in_ch=conv_ch * 2, out_ch=conv_ch * 2, kernel_size=(3, 21),
                                                padding=(1, 10), stride=(1, 2), dropout_coef=dropout_coef)
        out_shape = shape_change_conv(out_shape, kernel_size=(3, 21),
                                      padding=(1, 10), stride=(1, 2))
        self.conv4 = make_standard_double_layer(in_ch=conv_ch * 4, out_ch=conv_ch * 4, kernel_size=(1, 11),
                                                padding=(0, 5), stride=(1, 1), dropout_coef=dropout_coef)
        out_shape = shape_change_conv(out_shape, kernel_size=(3, 21),
                                      padding=(1, 10), stride=(1, 1))

        flat_dim = out_shape[0] * out_shape[1] * (conv_ch * 8)

        self.out_shape6 = flat_dim
        self.lin1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, padding=4, stride=1),
            nn.Flatten(),
            nn.Dropout(dropout_coef),
            nn.Linear(flat_dim, 30),
            nn.LeakyReLU(0.1),
            nn.Linear(30, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.lin1(x4)
        return x5


class ConvNet1(nn.Module):
    def __init__(self, input_shape=(12, 5000), n_classes=1):
        super(ConvNet1, self).__init__()
        '''
        This is basic convolutional neural network that is usually applied to computer vision tasks,
        but can be applied also in the sequence data analysis. In this neural network is also used
        residual connection to improve results.
        Parameters:
        -input shape - tuple (embedding size, sequence length). For ECG embedding size is number of predictors
        -n_classes - the number of classes that neural network fit to predict.
        '''
        # self.pos_enc = PositionalEncoder(input_shape[0], input_shape[1])

        out_shape = input_shape

        self.out_shape1 = out_shape

        dropout_coef = 0.3
        conv_ch = 24

        self.conv1 = nn.Sequential(
            nn.Dropout(dropout_coef),
            nn.Conv2d(in_channels=1, out_channels=conv_ch, kernel_size=(7, 101),
                      padding=(3, 30), stride=(2, 4)),
            nn.BatchNorm2d(conv_ch),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=5, padding=2, stride=1)
        )
        # int((H_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        out_shape = (int((out_shape[0] + 2 * 3 - 1 * (7 - 1) - 1) / 2 + 1),
                     int((out_shape[1] + 2 * 30 - 1 * (101 - 1) - 1) / 4 + 1))
        self.out_shape2 = out_shape

        self.conv2 = nn.Sequential(
            nn.Dropout(dropout_coef),
            nn.Conv2d(in_channels=conv_ch, out_channels=conv_ch * 2, kernel_size=(3, 41),
                      padding=(1, 20), stride=(1, 4)),
            nn.BatchNorm2d(conv_ch * 2),
            nn.MaxPool2d(kernel_size=5, padding=2, stride=1)
        )
        # int((H_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        out_shape = (out_shape[0],
                     int((out_shape[1] + 2 * 20 - 1 * (41 - 1) - 1) / 4 + 1))
        self.out_shape3 = out_shape

        self.conv3 = nn.Sequential(
            nn.Dropout(dropout_coef),
            nn.Conv2d(in_channels=conv_ch * 2, out_channels=conv_ch * 4, kernel_size=(3, 11),
                      padding=(1, 5), stride=(1, 4)),
            nn.BatchNorm2d(conv_ch * 4),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        )
        # int((H_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        out_shape = (out_shape[0],
                     int((out_shape[1] + 2 * 5 - 1 * (11 - 1) - 1) / 4 + 1))
        self.out_shape4 = out_shape

        self.conv4 = nn.Sequential(
            nn.Dropout(dropout_coef),
            nn.Conv2d(in_channels=conv_ch * 4, out_channels=conv_ch * 8, kernel_size=(1, 11),
                      padding=(0, 5), stride=(1, 4)),
            nn.BatchNorm2d(conv_ch * 8),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        )
        # int((H_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        out_shape = (out_shape[0],
                     int((out_shape[1] + 2 * 5 - 1 * (11 - 1) - 1) / 4 + 1))

        self.out_shape5 = out_shape

        flat_dim = out_shape[0] * out_shape[1] * (conv_ch * 8)

        self.out_shape6 = flat_dim
        self.lin1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=15, padding=7, stride=1),
            nn.Flatten(),
            nn.Dropout(dropout_coef),
            nn.Linear(flat_dim, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.lin1(x4)
        return x5

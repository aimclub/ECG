import torch
import torch.nn as nn

torch.manual_seed(42)

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        
        self.slope = 0.1

        self.conv1 = nn.Sequential(
            nn.Conv1d(12, 128, kernel_size=100),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=self.slope),
            nn.MaxPool1d(kernel_size=15)
        )

        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=50),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=self.slope),
            nn.MaxPool1d(kernel_size=15)
        )

        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(64, 64, kernel_size=20),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(negative_slope=self.slope),
        #     nn.MaxPool1d(kernel_size=5)
        # )

        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(64, 32, kernel_size=5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=self.slope),
            nn.MaxPool1d(kernel_size=2)
        )

        self.dense = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(2 * 32, 16)#,
            #nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )


    def forward_once(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        x = self.dense(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        out = self.classifier(torch.abs(out1 - out2))
        return out
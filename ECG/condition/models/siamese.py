import torch.nn as nn


class SiameseModel(nn.Module):

    def __init__(self,
                 kernel_size,
                 num_features,
                 like_LU_func,
                 norm1d,
                 dropout_rate
                 ):
        super(SiameseModel, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(12, num_features, kernel_size=kernel_size + 1),
            norm1d(num_features),
            like_LU_func()
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(num_features, num_features,
                      kernel_size=kernel_size + 2, padding=(kernel_size // 2), stride=1),
            norm1d(num_features),
            like_LU_func(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(num_features, num_features,
                      kernel_size=kernel_size + 2, padding=(kernel_size // 2), stride=4),
            norm1d(num_features),
            like_LU_func(),
            nn.Dropout(dropout_rate)
        )
        self.res1 = nn.Sequential(
            nn.MaxPool1d(4),
            nn.Conv1d(num_features, num_features, kernel_size=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(num_features, num_features * 2,
                      kernel_size=kernel_size + 2, padding=(kernel_size // 2), stride=1),
            norm1d(num_features * 2),
            like_LU_func(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(num_features * 2, num_features * 2,
                      kernel_size=kernel_size + 2, padding=(kernel_size // 2), stride=4),
            norm1d(num_features * 2),
            like_LU_func(),
            nn.Dropout(dropout_rate)
        )
        self.res2 = nn.Sequential(
            nn.MaxPool1d(4),
            nn.Conv1d(num_features, num_features * 2, kernel_size=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(num_features * 2, num_features * 2,
                      kernel_size=kernel_size + 2, padding=(kernel_size // 2), stride=1),
            norm1d(num_features * 2),
            like_LU_func(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(num_features * 2, num_features * 2,
                      kernel_size=kernel_size + 2, padding=(kernel_size // 2), stride=4),
            norm1d(num_features * 2),
            like_LU_func(),
            nn.Dropout(dropout_rate)
        )
        self.res3 = nn.Sequential(
            nn.MaxPool1d(4),
            nn.Conv1d(num_features * 2, num_features * 2, kernel_size=1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(num_features * 2, num_features * 3,
                      kernel_size=kernel_size + 2, padding=(kernel_size // 2), stride=1),
            norm1d(num_features * 3),
            like_LU_func(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(num_features * 3, num_features * 3,
                      kernel_size=kernel_size + 2, padding=(kernel_size // 2), stride=4),
            norm1d(num_features * 3),
            like_LU_func(),
            nn.Dropout(dropout_rate)
        )
        self.res4 = nn.Sequential(
            nn.MaxPool1d(4),
            nn.Conv1d(num_features * 2, num_features * 3, kernel_size=1)
        )

        feature_len = 4000 - kernel_size
        for _ in range(4):
            feature_len = feature_len // 4

        self.flatten = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=feature_len * num_features * 3, out_features=16),
            nn.Tanh()
        )

    def forward_once(self, x):

        x = self.conv0(x)

        x = self.conv1(x) + self.res1(x)

        x = self.conv2(x) + self.res2(x)

        x = self.conv3(x) + self.res3(x)

        x = self.conv4(x) + self.res4(x)

        x = self.flatten(x)

        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

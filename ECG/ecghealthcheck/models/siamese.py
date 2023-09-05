import torch
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size,
                 padding,
                 dropout_rate,
                 bn,
                 f_act
                 ):
        super().__init__()

        self.conv_l = nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size, padding=padding, stride=1),
            bn(out_features),
            f_act(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(out_features, out_features, kernel_size, padding=padding, stride=4),
            bn(out_features),
            f_act(),
            nn.Dropout(dropout_rate)
        )

        self.skip_con = nn.Sequential(
            nn.MaxPool1d(4),
            nn.Conv1d(in_features, out_features, kernel_size=1)
        )

        self.agg_conv = nn.Sequential(
            nn.Conv1d(out_features * 2, out_features, kernel_size=1),
            bn(out_features),
            f_act()
        )

    def forward(self, x):
        x = torch.cat((self.conv_l(x), self.skip_con(x)), dim=1)
        x = self.agg_conv(x)
        return x


class SiameseModel(nn.Module):

    def __init__(self,
                 kernel_size,
                 num_features,
                 like_LU_func,
                 norm1d,
                 dropout_rate,
                 n_res=4
                 ):
        super(SiameseModel, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(12, num_features, kernel_size=kernel_size + 1),
            norm1d(num_features),
            like_LU_func()
        )

        res_blocks = []
        for i in range(n_res):
            in_i = i // 2 + 1
            out_i = in_i + 1 if i % 2 == 1 else in_i
            res_blocks.append(
                ResBlock(
                    in_features=num_features * in_i,
                    out_features=num_features * out_i,
                    kernel_size=kernel_size + 2,
                    padding=kernel_size // 2,
                    dropout_rate=dropout_rate,
                    bn=norm1d,
                    f_act=like_LU_func
                )
            )
        self.res_blocks = nn.ModuleList(res_blocks)

        feature_len = 4000 - kernel_size
        for _ in range(n_res):
            feature_len = feature_len // 4

        self.flatten = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=feature_len * num_features * out_i, out_features=16),
            nn.Tanh()
        )

    def forward_once(self, x):

        x = self.conv0(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.flatten(x)

        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

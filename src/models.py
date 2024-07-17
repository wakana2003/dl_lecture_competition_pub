import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np

class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        super().__init__()
        self.W = nn.Parameter(torch.tensor(torch.randn(
            in_dim, out_dim
        ).normal_(0, torch.sqrt(torch.tensor(2.0)/in_dim)).float()))
        self.b = nn.Parameter(torch.zeros(out_dim).float())
        self.function = function

    def forward(self, x):
        return self.function(torch.matmul(x, self.W) + self.b)

# class Conv(nn.Module):
#     def __init__(self, filter_shape, function=lambda x: x, stride=(1, 1), padding=0):
#         super().__init__()
#         fan_in = filter_shape[1] * filter_shape[2] * filter_shape[3]
#         fan_out = filter_shape[0] * filter_shape[2] * filter_shape[3]

#         self.W = nn.Parameter(torch.tensor(torch.randn(
#             filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]
#         ).normal_(0, torch.sqrt(torch.tensor(2.0)/fan_in)).float()))

#         self.b = nn.Parameter(torch.zeros(filter_shape[0]).float())
#         self.function = function
#         self.stride = stride
#         self.padding = padding

#     def forward(self, x):
#         u = F.conv2d(x, self.W, bias=self.b, stride=self.stride, padding=self.padding)
#         return self.function(u)

 # nn.BatchNorm2d(hid_dim),
            # Dense(hid_dim * seq_len, hid_dim),
            # nn.ReLU(),

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.AvgPool1d(2),
            ConvBlock(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.AvgPool1d(2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
            nn.Dropout(p_drop)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)

class Dropout(nn.Module):
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.1):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x):
        # 学習時はdropout_ratio分だけ出力をシャットアウト
        if self.training:
            self.mask = torch.rand(*x.size()) > self.dropout_ratio
            return x * self.mask.to(x.device)
        # 推論時は出力に`1.0 - self.dropout_ratio`を乗算することで学習時の出力の大きさに合わせる
        else:
            return x * (1.0 - self.dropout_ratio)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        X = self.conv2(X) + X
        # X = F.glu(X, dim=-2)
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops.layers.torch import Rearrange



# class Pooling(nn.Module):
#     def __init__(self, ksize=(2, 2), stride=(2, 2), padding=0):
#         super().__init__()
#         self.ksize = ksize
#         self.stride = stride
#         self.padding = padding

#     def forward(self, x):
#         return F.avg_pool2d(x, kernel_size=self.ksize, stride=self.stride, padding=self.padding)

# class Flatten(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x.view(x.size()[0], -1)


class BatchNorm(nn.Module):
    def __init__(self, shape, epsilon=np.float32(1e-5)):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(np.ones(shape, dtype='float32')))
        self.beta = nn.Parameter(torch.tensor(np.zeros(shape, dtype='float32')))
        self.epsilon = epsilon

    def forward(self, x):
        mean = torch.mean(x, (0, 2, 3), keepdim=True)  # WRITE ME
        std = torch.std(x, (0, 2, 3), keepdim=True)  # WRITE ME
        x_normalized = (x - mean) / (std**2 + self.epsilon)**0.5  # WRITE ME
        return self.gamma * x_normalized + self.beta  # WRITE ME

class Activation(nn.Module):
    def __init__(self, function=lambda x: x):
        super().__init__()
        self.function = function

    def __call__(self, x):
        return self.function(x)

# class Flatten(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x.view(x.size()[0], -1)

# class BasicConvClassifier(nn.Module):
#     def __init__(
#         self,
#         num_classes: int,
#         seq_len: int,
#         in_channels: int,
#         hid_dim: int = 32
#     ) -> None:
#         super().__init__()

#         self.blocks = nn.Sequential(
#             Conv((hid_dim, in_channels, 3, 3), stride=(1, 1), padding=1),  # 入力チャネル数を考慮
#             BatchNorm((hid_dim, seq_len, seq_len)),
#             Activation(F.relu),
#             Pooling((2, 2)),  # サイズを適切に設定
#             Conv((hid_dim*2, hid_dim, 3, 3), stride=(1, 1), padding=1),
#             BatchNorm((hid_dim*2, seq_len//2, seq_len//2)),
#             Activation(F.relu),
#             Pooling((2, 2)),
#             Conv((hid_dim*4, hid_dim*2, 3, 3), stride=(1, 1), padding=1),
#             BatchNorm((hid_dim*4, seq_len//4, seq_len//4)),
#             Activation(F.relu),
#             Pooling((2, 2)),
#             Flatten(),
#             Dense(hid_dim*4*(seq_len//8)*(seq_len//8), 256, F.relu),
#             Dense(256, num_classes)
#         )

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         return self.blocks(X)

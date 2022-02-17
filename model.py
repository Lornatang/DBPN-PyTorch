# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
from torch import nn

__all__ = [
    "UpBlock", "DeepUpBlock", "DownBlock", "DeepDownBlock",
    "DBPN"
]


class UpBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super(UpBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv2 = torch.sub(conv2, identity)
        conv3 = self.conv3(conv2)

        out = torch.add(conv3, conv1)

        return out


class DeepUpBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int, padding: int, num_stage: int) -> None:
        super(DeepUpBlock, self).__init__()
        self.deep_conv = nn.Sequential(
            nn.Conv2d(channels * num_stage, channels, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deep_conv = self.deep_conv(x)
        conv1 = self.conv1(deep_conv)
        conv2 = self.conv2(conv1)
        conv2 = torch.sub(conv2, deep_conv)
        conv3 = self.conv3(conv2)

        out = torch.add(conv3, conv1)

        return out


class DownBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super(DownBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv2 = torch.sub(conv2, identity)
        conv3 = self.conv3(conv2)

        out = torch.add(conv3, conv1)

        return out


class DeepDownBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int, padding: int, num_stage: int) -> None:
        super(DeepDownBlock, self).__init__()
        self.deep_conv = nn.Sequential(
            nn.Conv2d(channels * num_stage, channels, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deep_conv = self.deep_conv(x)
        conv1 = self.conv1(deep_conv)
        conv2 = self.conv2(conv1)
        conv2 = torch.sub(conv2, deep_conv)
        conv3 = self.conv3(conv2)

        out = torch.add(conv3, conv1)

        return out


class DBPN(nn.Module):
    def __init__(self, upscale_factor: int) -> None:
        super(DBPN, self).__init__()
        if upscale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif upscale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif upscale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2
        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 256, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
        )

        # Second layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 64, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
        )

        # Back-projection stages layer
        self.up_block1 = UpBlock(64, kernel, stride, padding)
        self.down_block1 = DownBlock(64, kernel, stride, padding)
        self.up_block2 = UpBlock(64, kernel, stride, padding)

        self.deep_up_block1 = DeepUpBlock(64, kernel, stride, padding, num_stage=2)
        self.deep_up_block2 = DeepUpBlock(64, kernel, stride, padding, num_stage=3)
        self.deep_up_block3 = DeepUpBlock(64, kernel, stride, padding, num_stage=4)
        self.deep_up_block4 = DeepUpBlock(64, kernel, stride, padding, num_stage=5)
        self.deep_up_block5 = DeepUpBlock(64, kernel, stride, padding, num_stage=6)

        self.deep_down_block1 = DeepDownBlock(64, kernel, stride, padding, num_stage=2)
        self.deep_down_block2 = DeepDownBlock(64, kernel, stride, padding, num_stage=3)
        self.deep_down_block3 = DeepDownBlock(64, kernel, stride, padding, num_stage=4)
        self.deep_down_block4 = DeepDownBlock(64, kernel, stride, padding, num_stage=5)
        self.deep_down_block5 = DeepDownBlock(64, kernel, stride, padding, num_stage=6)

        # Final output layer
        self.conv3 = nn.Conv2d(192, 3, (3, 3), (1, 1), (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        out = []
        for _ in range(3):
            up_block1 = self.up_block1(conv2)
            down_block1 = self.down_block1(up_block1)
            up_block2 = self.up_block2(down_block1)

            concat1 = torch.cat([up_block2, up_block1], 1)
            deep_down_block1 = self.deep_down_block1(concat1)

            concat2 = torch.cat([deep_down_block1, down_block1], 1)
            deep_up_block1 = self.deep_up_block1(concat2)

            concat3 = torch.cat([deep_up_block1, concat1], 1)
            deep_down_block2 = self.deep_down_block2(concat3)

            concat4 = torch.cat([deep_down_block2, concat2], 1)
            deep_up_block2 = self.deep_up_block2(concat4)

            concat5 = torch.cat([deep_up_block2, concat3], 1)
            deep_down_block3 = self.deep_down_block3(concat5)

            concat6 = torch.cat([deep_down_block3, concat4], 1)
            deep_up_block3 = self.deep_up_block3(concat6)

            concat7 = torch.cat([deep_up_block3, concat5], 1)
            deep_down_block4 = self.deep_down_block4(concat7)

            concat8 = torch.cat([deep_down_block4, concat6], 1)
            deep_up_block4 = self.deep_up_block4(concat8)

            concat9 = torch.cat([deep_up_block4, concat7], 1)
            deep_down_block5 = self.deep_down_block5(concat9)

            concat10 = torch.cat([deep_down_block5, concat8], 1)
            deep_up_block5 = self.deep_up_block5(concat10)

            out.append(deep_up_block5)

        out = torch.cat(out, 1)

        out = self.conv3(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

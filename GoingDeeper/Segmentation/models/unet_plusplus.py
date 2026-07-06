import torch
import torch.nn as nn


class UNetPlusPlus(nn.Module):
    def __init__(self, input_channels=3, output_channels=1,
                 depth=4, base_filters=64, deep_supervision=False):
        """
        depth         : 다운샘플링(풀링) 단계 수. UNet 예시 코드와 동일한 구조는 depth=4
        base_filters  : 첫 conv 블록의 채널 수 (기본 64)
        deep_supervision : True이면 X_0,1 ~ X_0,depth 각각에서 출력 리스트 반환
        """
        super(UNetPlusPlus, self).__init__()

        self.depth = depth
        self.deep_supervision = deep_supervision
        nb_filter = [base_filters * (2 ** i) for i in range(depth + 1)]
        self.nb_filter = nb_filter

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.ModuleDict()

        # ---- Column 0 (Encoder backbone) : X_i,0 ----
        for i in range(depth + 1):
            in_ch = input_channels if i == 0 else nb_filter[i - 1]
            self.conv[f'conv{i}_0'] = self.double_conv(in_ch, nb_filter[i])

        # ---- Column j (Dense skip pathway) : X_i,j ----
        for j in range(1, depth + 1):
            for i in range(0, depth - j + 1):
                in_ch = nb_filter[i] * j + nb_filter[i + 1]
                self.conv[f'conv{i}_{j}'] = self.double_conv(in_ch, nb_filter[i])

        self.dropout = nn.Dropout(0.5)

        # ---- Output layer(s) ----
        if self.deep_supervision:
            self.finals = nn.ModuleList([
                nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
                for _ in range(depth)
            ])
        else:
            self.final = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        """2개의 Conv Layer로 이루어진 블록"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        depth = self.depth
        nodes = {}

        # ---- Column 0 (Encoder) ----
        nodes['0_0'] = self.conv['conv0_0'](x)
        for i in range(1, depth + 1):
            out = self.conv[f'conv{i}_0'](self.pool(nodes[f'{i - 1}_0']))
            if i == depth:
                out = self.dropout(out)  # bottleneck에 dropout 적용
            nodes[f'{i}_0'] = out

        # ---- Column 1..depth (Dense skip pathway) ----
        for j in range(1, depth + 1):
            for i in range(0, depth - j + 1):
                same_row = [nodes[f'{i}_{k}'] for k in range(j)]
                up = self.up(nodes[f'{i + 1}_{j - 1}'])
                cat = torch.cat(same_row + [up], dim=1)
                nodes[f'{i}_{j}'] = self.conv[f'conv{i}_{j}'](cat)

        # ---- Output ----
        if self.deep_supervision:
            outputs = []
            for j in range(1, depth + 1):
                outputs.append(torch.sigmoid(self.finals[j - 1](nodes[f'0_{j}'])))
            return outputs
        else:
            return torch.sigmoid(self.final(nodes[f'0_{depth}']))
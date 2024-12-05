import torch
from torch import nn
from torch.nn import functional as F
from typing import Type

class LoRALinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int,  r, scale, bias: bool=True) -> None:
        super().__init__(in_features, out_features, bias)
        self.r = r
        self.trainable_lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.trainable_lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()

        nn.init.normal_(self.trainable_lora_down.weight, std=1/r)
        nn.init.zeros_(self.trainable_lora_up.weight)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        lora_adjustment = self.scale*self.dropout(self.trainable_lora_up(self.selector(self.trainable_lora_down(x))))
        out = out + lora_adjustment
        return  out

class LoRALinearTwo(nn.Linear):
    def __init__(self, in_features: int, out_features: int,  r, scale, bias: bool=True) -> None:
        super().__init__(in_features, out_features, bias)
        self.r = r
        self.trainable_lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.trainable_lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()

        nn.init.normal_(self.trainable_lora_down.weight, std=1/r)
        nn.init.zeros_(self.trainable_lora_up.weight)

        self.r2 = r
        self.trainable_lora_down2 = nn.Linear(in_features, r, bias=False)
        self.dropout2 = nn.Dropout(0.1)
        self.trainable_lora_up2 = nn.Linear(r, out_features, bias=False)
        self.scale2 = scale
        self.selector2 = nn.Identity()

        nn.init.normal_(self.trainable_lora_down2.weight, std=1/self.r2)
        nn.init.zeros_(self.trainable_lora_up2.weight)

    def forward(self, x, alpha):
        out = F.linear(x, self.weight, self.bias)
        lora_adjustment_1 = self.scale*self.dropout(self.trainable_lora_up(self.selector(self.trainable_lora_down((1-alpha)*x))))
        lora_adjustment_2 = self.scale2*self.dropout2(self.trainable_lora_up2(self.selector2(self.trainable_lora_down2(alpha*x))))
        out = out + lora_adjustment_1 + lora_adjustment_2
        return  out

class LoRAConv2D(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, r, scale, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) -> None:
        if isinstance(padding, int):
            padding = (padding, padding) 
        if isinstance(dilation, int):
            dilation = (dilation, dilation) 
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        assert type(kernel_size) is int
        self.r = r
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.trainable_lora_down = nn.Conv2d(
            in_channels = in_channels,
            out_channels = r,
            kernel_size = kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )

        self.trainable_lora_up = nn.Conv2d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        self.selector = nn.Identity()
        self.scale = scale

        nn.init.normal_(self.trainable_lora_down.weight, std=1/r)
        nn.init.zeros_(self.trainable_lora_up.weight)

    def forward(self, x):
        out = F.conv2d(x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        lora_adjustment = self.scale*self.trainable_lora_up(self.selector(self.trainable_lora_down(x)))
        out = out + lora_adjustment
        return out
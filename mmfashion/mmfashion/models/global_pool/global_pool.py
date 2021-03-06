from __future__ import division

import torch.nn as nn

from ..registry import GLOBALPOOLING


@GLOBALPOOLING.register_module
class GlobalPooling(nn.Module):

    def __init__(self, inplanes, pool_plane, inter_channels, outchannels):
        super(GlobalPooling, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(inplanes)

        inter_plane = inter_channels[0] * inplanes[0] * inplanes[1]

        self.global_layers = nn.Sequential(
            nn.Linear(inter_plane, inter_channels[1]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(inter_channels[1], outchannels),
            nn.ReLU(True),
            nn.Dropout(),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        global_pool = self.global_layers(x)
        return global_pool

    def init_weights(self):
        if isinstance(self.global_layers, nn.Sequential):
            for m in self.global_layers:
                if type(m) == nn.Linear:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.01)
        else:
            self.global_layers.init_weights()

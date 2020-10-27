import torch
from torch import nn
import torch.nn.functional as F

from lib import layers


class BaseASPPNet(nn.Module):

    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        self.aspp = layers.ASPPModule(ch * 8, ch * 8, dilations)

        self.dec4 = layers.Decoder(ch * (8 + 8), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        h = self.aspp(h)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        return h


class CascadedASPPNet(nn.Module):

    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        self.stg1_low_band_net = BaseASPPNet(4, 16)
        self.stg1_high_band_net = BaseASPPNet(4, 16)

        self.stg2_bridge = layers.Conv2DBNActiv(20, 8, 1, 1, 0)
        self.stg2_low_band_net = BaseASPPNet(8, 16)
        self.stg2_high_band_net = BaseASPPNet(8, 16)

        self.stg3_mag_bridge = layers.Conv2DBNActiv(34, 8, 1, 1, 0)
        self.stg3_full_band_net = BaseASPPNet(8, 16)

        self.stg3_phase_bridge = layers.Conv2DBNActiv(34, 8, 1, 1, 0)
        self.stg3_phase_net = BaseASPPNet(8, 16)

        self.out = nn.Conv2d(16, 2, 1, bias=False)
        self.phase_out = nn.Conv2d(16, 2, 1, bias=False)
        self.aux1_out = nn.Conv2d(16, 2, 1, bias=False)
        self.aux2_out = nn.Conv2d(16, 2, 1, bias=False)

        self.max_bin = n_fft // 2
        self.band_size = self.max_bin // 2
        self.output_bin = n_fft // 2 + 1

        self.offset = 128

    def forward(self, x_mag, x_phase):
        x = torch.cat([x_mag, x_phase], dim=1)
        x = x[:, :, :self.max_bin]

        h_stg1_low = self.stg1_low_band_net(x[:, :, :self.band_size])
        h_stg1_high = self.stg1_high_band_net(x[:, :, self.band_size:])
        aux1 = torch.cat([h_stg1_low, h_stg1_high], dim=2)

        h = torch.cat([x, aux1], dim=1)
        h = self.stg2_bridge(h)
        h_stg2_low = self.stg2_low_band_net(h[:, :, :self.band_size])
        h_stg2_high = self.stg2_high_band_net(h[:, :, self.band_size:])
        aux2 = torch.cat([h_stg2_low, h_stg2_high], dim=2)

        h = torch.cat([x[:, :2], aux1, aux2], dim=1)
        h = self.stg3_mag_bridge(h)
        h = self.stg3_full_band_net(h)

        mask = torch.sigmoid(self.out(h))
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode='replicate')

        h = torch.cat([x[:, 2:, :self.band_size], h_stg1_low, h_stg2_low], dim=1)
        h = self.stg3_phase_bridge(h)
        h = self.stg3_phase_net(h)

        phase_mask = torch.tanh(self.phase_out(h))
        phase_mask = torch.cat([phase_mask, torch.ones_like(x_phase[:, :, self.band_size:])], dim=2)

        if self.training:
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            aux1 = F.pad(
                input=aux1,
                pad=(0, 0, 0, self.output_bin - aux1.size()[2]),
                mode='replicate')
            aux2 = torch.sigmoid(self.aux2_out(aux2))
            aux2 = F.pad(
                input=aux2,
                pad=(0, 0, 0, self.output_bin - aux2.size()[2]),
                mode='replicate')
            return mask, phase_mask, aux1, aux2
        else:
            return mask, phase_mask

    def predict(self, x_mag, x_phase):
        mask, phase_mask = self.forward(x_mag, x_phase)

        mag = mask * x_mag
        phase_mask[mask > 0.5] = 1
        phase = phase_mask * x_phase

        if self.offset > 0:
            mag = mag[:, :, :, self.offset:-self.offset]
            phase = phase[:, :, :, self.offset:-self.offset]
            assert mag.size()[3] > 0 and phase.size()[3] > 0

        return mag, phase

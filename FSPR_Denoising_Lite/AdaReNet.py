import torch
import torch.nn as nn
import FCNN_plus as fn


class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv2, pool2
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv3, pool3
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv4, pool4
        self._block4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv5, pool5
        self._block5 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv6, upsample5
        self._block6 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block7 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv4a, dec_deconv4b, upsample3
        self._block8 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        
        # Layers: dec_deconv3a, dec_deconv3b, upsample2
        self._block9 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        
        # Layers: dec_deconv2a, dec_deconv2b, upsample1
        self._block10 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block11 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1))

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block3(pool2)
        pool4 = self._block4(pool3)
        pool5 = self._block5(pool4)

        # Decoder
        upsample5 = self._block6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block7(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block8(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block9(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block10(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block11(concat1)
      
class UNet_F(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet_F, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            fn.Fconv_PCA(3, in_channels, 12, 4, inP=3, ifIni=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1), 
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv2, pool2
        self._block2 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv3, pool3
        self._block3 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv4, pool4
        self._block4 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv5, pool5
        self._block5 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2))
        
        # Layers: enc_conv6, upsample5
        self._block6 = nn.Sequential(
            fn.Fconv_PCA(3, 12, 12, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block7 = nn.Sequential(
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv4a, dec_deconv4b, upsample3
        self._block8 = nn.Sequential(
            fn.Fconv_PCA(3, 36, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        
        # Layers: dec_deconv3a, dec_deconv3b, upsample2
        self._block9 = nn.Sequential(
            fn.Fconv_PCA(3, 36, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))
        
        # Layers: dec_deconv2a, dec_deconv2b, upsample1
        self._block10 = nn.Sequential(
            fn.Fconv_PCA(3, 36, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA(3, 24, 24, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block11 = nn.Sequential(
            fn.Fconv_PCA(3, 96 + in_channels, 16, 4, inP=3, ifIni=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA(3, 16, 8, 4, inP=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.1, inplace=True),
            fn.Fconv_PCA_out(3, 8, out_channels, 4, inP=3, padding=1))
        

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block3(pool2)
        pool4 = self._block4(pool3)
        pool5 = self._block5(pool4)

        # Decoder
        upsample5 = self._block6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block7(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block8(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block9(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block10(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block11(concat1)
    

def local_variance_map(x, kernel_size=5):
    """
    Local noise variance estimation M_noise = V(I).
    Differentiable: local variance in k×k window.
    Returns (B, 1, H, W) noise map.
    """
    padding = kernel_size // 2
    x_mean = nn.functional.avg_pool2d(x, kernel_size, stride=1, padding=padding)
    x_sq_mean = nn.functional.avg_pool2d(x ** 2, kernel_size, stride=1, padding=padding)
    var = x_sq_mean - x_mean ** 2
    var = torch.clamp(var, min=1e-8)  # numerical stability
    # Average across channels -> (B, 1, H, W)
    m_noise = var.mean(dim=1, keepdim=True)
    return m_noise


class MaskNetwork(nn.Module):
    """Original mask network (vanilla convolutions)."""
    def __init__(self, in_channels, out_channels):
        super(MaskNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        mask = self.sigmoid(x) 
        return mask


class RealFrequencyModule_M1(nn.Module):
    """
    相位保留频域滤波 (Phase-Preserving FFT).
    FFT → 幅度/相位分离 → 仅对幅度施加可学习掩码 → 原相位重建 → IFFT → 残差。
    相位 spectrum 不做任何学习或改动。
    """
    def __init__(self, channels):
        super(RealFrequencyModule_M1, self).__init__()
        # 轻量 1x1 分支：学习如何抑制幅度谱中的噪声（输出 [0,1] 掩码）
        self.amp_filter = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, viz_dict=None):
        fft_x = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        if viz_dict is not None:
            viz_dict['m1_amp_before'] = amp.detach()
        amp_mask = self.amp_filter(amp)
        clean_amp = amp * amp_mask
        if viz_dict is not None:
            viz_dict['m1_amp_after'] = clean_amp.detach()
        fft_clean = clean_amp * torch.exp(1j * phase)
        out = torch.fft.ifftn(fft_clean, dim=(-2, -1), norm='ortho').real
        return out + x


class ResNetBlock(nn.Module):
    """Original simple ResNet block for self-correction."""
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)                 
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, out_channels, 3, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x + residual 
        return x


class MSSAB(nn.Module):
    """
    Multi-Scale Spatial Attention Block (多尺度空间注意力块).
    Advanced Self-Correcting Module - replaces simple ResNet with larger receptive field.
    
    Step 1: F1,F2,F3 = DConv_d(Î) for d=1,2,4
    Step 2: F_multi = Conv_1x1([F1,F2,F3])
    Step 3: W_att = σ(Conv_7x7(F_multi)), Ī = Î + W_att ⊙ F_multi
    """
    def __init__(self, in_channels, out_channels, mid_channels=32):
        super(MSSAB, self).__init__()
        
        # Step 1: Multi-scale dilated convolutions (d=1, 2, 4)
        self.dconv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.dconv4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
        )
        
        # Step 2: Feature aggregation - 1x1 conv
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels, 1),
            nn.ReLU(inplace=True),
        )
        
        # Step 3: Gated spatial attention - 7x7 conv for local context
        self.att_conv = nn.Sequential(
            nn.Conv2d(mid_channels, 1, 7, padding=3),
            nn.Sigmoid(),
        )
        
        # Project back to output channels
        self.proj = nn.Conv2d(mid_channels, out_channels, 1)
    
    def forward(self, x, viz_dict=None):
        residual = x  # Î
        
        # Step 1: Multi-scale feature extraction
        f1 = self.dconv1(x)
        f2 = self.dconv2(x)
        f3 = self.dconv4(x)
        
        # Step 2: Feature aggregation
        f_multi = torch.cat([f1, f2, f3], dim=1)
        f_multi = self.conv_fuse(f_multi)
        
        # Step 3: Gated attention correction（att_conv 末层含 Sigmoid，w_att 即注意力图）
        w_att = self.att_conv(f_multi)
        if viz_dict is not None:
            viz_dict['m2_attention'] = w_att.detach()
        f_corrected = self.proj(f_multi)
        
        # Ī = Î + W_att ⊙ F_multi (residual + gated correction)
        out = residual + w_att * f_corrected
        return out


class DynamicGate(nn.Module):
    """
    动态门控：Output = x + α · module(x)，α 为可学习标量，初值 0。
    Epoch 0 时等价于恒等映射（纯骨干路径），训练中由反向传播自适应调节各分支强度。
    """
    def __init__(self, module):
        super(DynamicGate, self).__init__()
        self.module = module
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, viz_dict=None):
        y = self.module(x, viz_dict) if viz_dict is not None else self.module(x)
        return x + self.alpha * y


class AdaReNet(nn.Module):
    """
    论文五档实验（ada_config，与 checkpoint 目录 c0~c4 一致）:
    - 0: 纯骨干（无双 UNet 前置 M1/M2）
    - 1: 硬串联 M1→M2，无 DynamicGate（对照/负例）
    - 2: 仅 M1 + DynamicGate（α 初值 0）
    - 3: 仅 M2 + DynamicGate
    - 4: 与 c1 相同 FSPR（M1→M2 硬串联）；推理时配合 Test-Time Augmentation（见 utils.tta_forward）
    已移除 M1∥M2 并联与 dual_domain_fusion。
    """
    def __init__(self, in_channels=3, out_channels=3, ada_config=0):
        super(AdaReNet, self).__init__()
        assert ada_config in (0, 1, 2, 3, 4), f'ada_config must be 0~4, got {ada_config}'
        self.ada_config = ada_config

        self.unet = UNet(in_channels, out_channels)
        self.unet_f = UNet_F(in_channels, out_channels)
        self.mask_network = MaskNetwork(in_channels, out_channels=1)

        if ada_config in (1, 4):
            self.real_freq_m1 = RealFrequencyModule_M1(in_channels)
            # MSSAB 残差与 x 同通道数，须映射回 in_channels（不能与 out_channels 混用）
            self.mssab_m2 = MSSAB(in_channels, in_channels)
        elif ada_config == 2:
            self.real_freq_m1 = DynamicGate(RealFrequencyModule_M1(in_channels))
        elif ada_config == 3:
            self.mssab_m2 = DynamicGate(MSSAB(in_channels, in_channels))

        # 接在融合后的 UNet 输出上，输入为 out_channels
        self.self_correct = ResNetBlock(out_channels, out_channels)

    def forward(self, x, viz_dict=None):
        feat = x
        ac = self.ada_config
        if ac == 1 or ac == 4:
            feat = self.real_freq_m1(feat, viz_dict)
            feat = self.mssab_m2(feat, viz_dict)
        elif ac == 2:
            feat = self.real_freq_m1(feat, viz_dict)
        elif ac == 3:
            feat = self.mssab_m2(feat, viz_dict)

        output_unet = self.unet(feat)
        output_unet_f = self.unet_f(feat)
        mask = self.mask_network(feat)
        combined_output = output_unet * mask + output_unet_f * (1 - mask)
        return self.self_correct(combined_output)
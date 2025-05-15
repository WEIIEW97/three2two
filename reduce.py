import torch
import torch.nn as nn
import torch.nn.functional as F


class ReducedConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
        bias=False,
    ):
        super(ReducedConv3d, self).__init__()

        self.C_in = in_channels
        self.C_out = out_channels
        self.K_d, self.K_h, self.K_w = kernel_size
        self.S_d, self.S_h, self.S_w = stride
        self.P_d, self.P_h, self.P_w = padding
        self.Dil_d, self.Dil_h, self.Dil_w = 1, 1, 1
        self.groups = 1
        self.bias = bias

        self.conv2d = nn.Conv2d(
            in_channels=self.C_in * self.K_d,
            out_channels=self.C_out,
            kernel_size=(self.K_h, self.K_w),
            stride=(self.S_h, self.S_w),
            padding=(self.P_h, self.P_w),
            dilation=(self.Dil_h, self.Dil_w),
            groups=self.groups,
            bias=self.bias,
        )

    def transplant_weights_from_conv3d(self, conv3d_layer: nn.Module):
        if not isinstance(conv3d_layer, nn.Conv3d):
            raise TypeError("Input layer must be an instance of nn.Conv3d")

        if conv3d_layer.dilation[0] != 1:  # depth dilation must be 1
            raise ValueError(
                "Conv3dMimic only supports nn.Conv3d with depth dilation (dilation[0]) of 1."
            )

        self.C_in = conv3d_layer.in_channels
        self.C_out = conv3d_layer.out_channels
        self.K_d, self.K_h, self.K_w = conv3d_layer.kernel_size
        self.S_d, self.S_h, self.S_w = conv3d_layer.stride
        self.P_d, self.P_h, self.P_w = conv3d_layer.padding
        self.Dil_h, self.Dil_w = (
            conv3d_layer.dilation[1],
            conv3d_layer.dilation[2],
        )  # depth dilation is 1
        self.groups = conv3d_layer.groups
        self.bias = conv3d_layer.bias is not None

        self.conv2d = nn.Conv2d(
            in_channels=self.C_in * self.K_d,
            out_channels=self.C_out,
            kernel_size=(self.K_h, self.K_w),
            stride=(self.S_h, self.S_w),
            padding=(self.P_h, self.P_w),
            dilation=(self.Dil_h, self.Dil_w),
            groups=self.groups,
            bias=self.bias,
        )

        with torch.no_grad():
            # original 3D weights: (C_out, C_in / groups, K_d, K_h, K_w)
            # reshaped 2D weights: (C_out, (C_in * K_d) / groups, K_h, K_w)
            # (C_in * K_d) / groups == (C_in / groups) * K_d
            weights_2d = conv3d_layer.weight.data.clone().view(
                self.C_out, (self.C_in // self.groups) * self.K_d, self.K_h, self.K_w
            )
            self.conv2d.weight.data = weights_2d

            if self.bias:
                self.conv2d.bias.data = conv3d_layer.bias.data.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, D, H, W = x.shape
        if C != self.C_in:
            raise ValueError(
                f"Input channel mismatch in Conv3dMimic ({self.C_in=}, {self.K_d=}): "
                f"got {C}, expected {self.C_in}"
            )

        xp = x
        if self.P_d > 0:
            xp = F.pad(xp, (0, 0, 0, 0, self.P_d, self.P_d), mode="constant", value=0)

        D_padded = xp.shape[2]
        # If input depth after padding is less than kernel depth, unfold won't work or produce empty.
        # Conv3d itself would error or produce specific output sizes.
        # For unfold: size (K_d) must be less than or equal to dimension size (D_padded)
        if D_padded < self.K_d:
            # Calculate expected output shape as Conv3d would (likely empty or error)
            # D_out = floor((D_padded - K_d)/S_d + 1) -> will be <= 0
            # H_out = floor((H_in + 2*P_h - Dil_h*(K_h-1) - 1)/S_h + 1)
            # W_out = floor((W_in + 2*P_w - Dil_w*(K_w-1) - 1)/S_w + 1)
            # For simplicity, we'll let unfold handle it or error if input is too small,
            # similar to how actual Conv3D might behave.
            # A truly robust version might pre-calculate output dimensions.
            # If D_out_calc becomes 0, we'll return an empty tensor later.
            pass

        # Unfold along the depth dimension: extracts sliding blocks
        # .unfold(dimension, size, step)
        x_unfold = xp.unfold(
            2, self.K_d, self.S_d
        )  # (N, C_in, D_out_calc, H_in_actual, W_in_actual, K_d)
        D_out_calc = x_unfold.shape[2]

        # (N, C_in, D_out_calc, H_actual, W_actual, K_d) -> (N, D_out_calc, C_in, K_d, H_actual, W_actual)
        x_perm = x_unfold.permute(0, 2, 1, 5, 3, 4)

        N_eff = N * D_out_calc
        C_eff = self.C_in * self.K_d

        H_conv2d_in = x_perm.shape[4]
        W_conv2d_in = x_perm.shape[5]

        if N_eff == 0:
            H_out_final = (
                H + 2 * self.P_h - self.Dil_h * (self.K_h - 1) - 1
            ) // self.S_h + 1
            W_out_final = (
                W + 2 * self.P_w - self.Dil_w * (self.K_w - 1) - 1
            ) // self.S_w + 1
            return torch.empty(
                (N, self.C_out, H_out_final, W_out_final),
                device=x.device,
                dtype=x.dtype,
            )

        x_conv2d_in = x_perm.contiguous().view(N_eff, C_eff, H_conv2d_in, W_conv2d_in)
        y_conv2d_out = self.conv2d(x_conv2d_in)
        _N_eff_out, _C_out_out, H_out_2d, W_out_2d = y_conv2d_out.shape
        y_reshaped = y_conv2d_out.view(N, D_out_calc, self.C_out, H_out_2d, W_out_2d)
        y = y_reshaped.permute(0, 2, 1, 3, 4)
        return y


class ReducedBatchNorm3d(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(ReducedBatchNorm3d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

        self.bn2d = nn.BatchNorm2d(
            num_features=self.num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )

    def transplant_weights_from_bn3d(self, bn3d: nn.Module):
        if not isinstance(bn3d, nn.BatchNorm3d):
            raise TypeError("Input layer must be an instance of nn.BatchNorm3d")

        self.num_features = bn3d.num_features
        self.eps = bn3d.eps
        self.momentum = bn3d.momentum
        self.affine = bn3d.affine
        self.track_running_stats = bn3d.track_running_stats

        self.bn2d = nn.BatchNorm2d(
            num_features=self.num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )

        # Transplant weights, bias, running_mean, running_var
        with torch.no_grad():
            if self.affine:
                if bn3d.weight is not None:
                    self.bn2d.weight.data = bn3d.weight.data.clone()
                if bn3d.bias is not None:
                    self.bn2d.bias.data = bn3d.bias.data.clone()

            if self.track_running_stats:
                if bn3d.running_mean is not None:
                    self.bn2d.running_mean.data = bn3d.running_mean.data.clone()
                if bn3d.running_var is not None:
                    self.bn2d.running_var.data = bn3d.running_var.data.clone()
                if bn3d.num_batches_tracked is not None:
                    self.bn2d.num_batches_tracked.data = (
                        bn3d.num_batches_tracked.data.clone()
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, D, H, W = x.shape
        if C != self.num_features:
            raise ValueError(
                f"Channel mismatch in BatchNorm3dMimic ({self.num_features=}): got {C}"
            )

        if D == 0:
            return x

        x_reshape = x.permute(0, 2, 1, 3, 4).contiguous().view(N * D, C, H, W)

        bn_out = self.bn2d(x_reshape)
        # (N*D, C, H, W) -> (N, D, C, H, W) -> (N, C, D, H, W)
        out = bn_out.view(N, D, C, H, W).permute(0, 2, 1, 3, 4)
        return out

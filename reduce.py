import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


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


def interpolate_3d(x: torch.Tensor, size: tuple | list) -> torch.Tensor:
    """
    Interpolates a 3D tensor to a new size using bilinear interpolation.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C, D, H, W).
        size (tuple | list): New size for the output tensor (D_out, H_out, W_out).

    Returns:
        torch.Tensor: Interpolated tensor of shape (N, C, D_out, H_out, W_out).
    """
    N, C, D, H, W = x.shape
    if len(size) != 3:
        raise ValueError("Size must be a tuple or list of length 3.")

    raise NotImplementedError


def bilinear_grid_sample(
    im: torch.Tensor,
    grid: torch.Tensor,
    align_corners: bool = False,
    padding_mode: str = "constant",
) -> torch.Tensor:
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners (bool): If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input's
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input's corner pixels,
            making the sampling more resolution agnostic.

    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode=padding_mode, value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)


def bi_to_tri_v1(
    x: torch.Tensor,
    grid: torch.Tensor,
    align_corners: bool = True,
    padding_mode: str = "border",
) -> torch.Tensor:
    N, C, D_in, H_in, W_in = x.shape
    _, D_out, H_out, W_out = grid.shape

    xy_coords = grid[..., 0:2]
    z_coords = grid[..., 2]

    if align_corners:
        z_denorm = (z_coords + 1) / 2 * (D_in - 1)
    else:
        z_denorm = ((z_coords + 1) / 2 * D_in) - 0.5

    z_floor_idx = torch.floor(z_denorm)
    z_ceil_idx = torch.ceil(z_denorm)

    z_weights = z_denorm - z_floor_idx
    z_weights = torch.clamp(z_weights, 0.0, 1.0)
    z_weights = z_weights.unsqueeze(1)

    z_floor_idx_clamped = torch.clamp(z_floor_idx.long(), 0, D_in - 1)
    z_ceil_idx_clamped = torch.clamp(z_ceil_idx.long(), 0, D_in - 1)

    K = D_out * H_out * W_out
    batch_indices_for_gather = torch.arange(N, device=x.device).repeat_interleave(K)

    z_floor_flat = z_floor_idx_clamped.reshape(N * K)
    z_ceil_flat = z_ceil_idx_clamped.reshape(N * K)

    slices_floor = x[batch_indices_for_gather, :, z_floor_flat, :, :]
    slices_ceil = x[batch_indices_for_gather, :, z_ceil_flat, :, :]

    xy_coords_for_gs = xy_coords.contiguous().view(N * K, 1, 1, 2)

    interp_val_floor = F.grid_sample(
        slices_floor,
        xy_coords_for_gs,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    interp_val_ceil = F.grid_sample(
        slices_ceil,
        xy_coords_for_gs,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    interp_val_floor = interp_val_floor.view(N, D_out, H_out, W_out, C).permute(
        0, 4, 1, 2, 3
    )
    interp_val_ceil = interp_val_ceil.view(N, D_out, H_out, W_out, C).permute(
        0, 4, 1, 2, 3
    )

    output = (1.0 - z_weights) * interp_val_floor + z_weights * interp_val_ceil
    return output


# onnx export compatible version, for which opset version <= 11
def bi_to_tri_v2(
    x: torch.Tensor,
    grid: torch.Tensor,
    align_corners: bool = True,
) -> torch.Tensor:

    N, C, D_in, H_in, W_in = x.shape
    _, D_out, H_out, W_out = grid.shape

    xy_coords = grid[..., 0:2]
    z_coords = grid[..., 2]

    if align_corners:
        z_denorm = (z_coords + 1) / 2 * (D_in - 1)
    else:
        z_denorm = ((z_coords + 1) / 2 * D_in) - 0.5

    z_floor_idx = torch.floor(z_denorm)
    z_ceil_idx = torch.ceil(z_denorm)

    z_weights = z_denorm - z_floor_idx
    z_weights = torch.clamp(z_weights, 0.0, 1.0)
    z_weights = z_weights.unsqueeze(1)

    z_floor_idx_clamped = torch.clamp(z_floor_idx.long(), 0, D_in - 1)
    z_ceil_idx_clamped = torch.clamp(z_ceil_idx.long(), 0, D_in - 1)

    K = D_out * H_out * W_out
    batch_indices_for_gather = torch.arange(N, device=x.device).repeat_interleave(K)

    z_floor_flat = z_floor_idx_clamped.reshape(N * K)
    z_ceil_flat = z_ceil_idx_clamped.reshape(N * K)

    slices_floor = x[batch_indices_for_gather, :, z_floor_flat, :, :]
    slices_ceil = x[batch_indices_for_gather, :, z_ceil_flat, :, :]

    xy_coords_for_gs = xy_coords.contiguous().view(N * K, 1, 1, 2)

    interp_val_floor = bilinear_grid_sample(
        slices_floor,
        xy_coords_for_gs,
        align_corners=align_corners,
    )
    interp_val_ceil = bilinear_grid_sample(
        slices_ceil,
        xy_coords_for_gs,
        align_corners=align_corners,
    )

    interp_val_floor = interp_val_floor.view(N, D_out, H_out, W_out, C).permute(
        0, 4, 1, 2, 3
    )
    interp_val_ceil = interp_val_ceil.view(N, D_out, H_out, W_out, C).permute(
        0, 4, 1, 2, 3
    )

    output = (1.0 - z_weights) * interp_val_floor + z_weights * interp_val_ceil
    return output


def bi_to_tri(
    x: torch.Tensor,
    grid: torch.Tensor,
    align_corners: bool = True,
    export_compatible: bool = True,
) -> torch.Tensor:
    return (
        bi_to_tri_v2(x, grid, align_corners)
        if export_compatible
        else bi_to_tri_v1(x, grid, align_corners)
    )


def _generate_grid(
    output_size: Tuple[int, int, int],
    align_corners: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Generates a uniform sampling grid for a given output size,
    matching the conventions of F.interpolate.
    Returns a grid of shape (D_out, H_out, W_out, 3).
    """
    D_out, H_out, W_out = output_size

    if align_corners:
        grid_z_1d = (
            torch.linspace(-1.0, 1.0, D_out, device=device, dtype=dtype)
            if D_out > 1
            else torch.tensor([0.0], device=device, dtype=dtype)
        )
        grid_y_1d = (
            torch.linspace(-1.0, 1.0, H_out, device=device, dtype=dtype)
            if H_out > 1
            else torch.tensor([0.0], device=device, dtype=dtype)
        )
        grid_x_1d = (
            torch.linspace(-1.0, 1.0, W_out, device=device, dtype=dtype)
            if W_out > 1
            else torch.tensor([0.0], device=device, dtype=dtype)
        )
    else:
        # Generates coordinates for pixel centers, scaled to [-1, 1]
        # For Dim_out=1, coord is 0.5, normalized is 0.0
        # For Dim_out=2, coords are 0.25, 0.75, normalized are -0.5, 0.5
        grid_z_1d = (
            (torch.arange(D_out, device=device, dtype=dtype) + 0.5) / D_out * 2.0 - 1.0
            if D_out > 0
            else torch.empty(0, device=device, dtype=dtype)
        )
        grid_y_1d = (
            (torch.arange(H_out, device=device, dtype=dtype) + 0.5) / H_out * 2.0 - 1.0
            if H_out > 0
            else torch.empty(0, device=device, dtype=dtype)
        )
        grid_x_1d = (
            (torch.arange(W_out, device=device, dtype=dtype) + 0.5) / W_out * 2.0 - 1.0
            if W_out > 0
            else torch.empty(0, device=device, dtype=dtype)
        )

        # Handle D_out/H_out/W_out = 1 case explicitly for arange logic to be consistent
        if D_out == 1:
            grid_z_1d = torch.tensor([0.0], device=device, dtype=dtype)
        if H_out == 1:
            grid_y_1d = torch.tensor([0.0], device=device, dtype=dtype)
        if W_out == 1:
            grid_x_1d = torch.tensor([0.0], device=device, dtype=dtype)

    mesh_z, mesh_y, mesh_x = torch.meshgrid(
        grid_z_1d, grid_y_1d, grid_x_1d, indexing="ij"
    )
    # Stack in (x, y, z) order for grid_sample compatibility
    grid = torch.stack((mesh_x, mesh_y, mesh_z), dim=-1)
    return grid


def interpolate_3d(
    x: torch.Tensor,
    size: tuple,
    align_corners: bool = True,
    export_compatible: bool = True,
) -> torch.Tensor:
    """
    Resizes a 3D volume using mimicked trilinear interpolation (based on bilinear ops).
    This function generates an implicit grid similar to F.interpolate.

    Args:
        volume (torch.Tensor): The input 3D volume of shape (N, C, D_in, H_in, W_in).
        output_size (Tuple[int, int, int]): The target output spatial size (D_out, H_out, W_out).
        align_corners (bool): Argument for grid generation and for the underlying F.grid_sample calls.

    Returns:
        torch.Tensor: The resized output of shape (N, C, D_out, H_out, W_out).
    """
    N, C, _, _, _ = x.shape  # D_in, H_in, W_in not directly used for grid gen here
    device = x.device
    dtype = x.dtype  # Generate grid with same dtype as volume's float components

    if not (isinstance(size, tuple) and len(size) == 3):
        raise ValueError(
            "output_size must be a tuple of 3 integers (D_out, H_out, W_out)"
        )
    if not all(isinstance(s, int) and s > 0 for s in size):
        raise ValueError("All dimensions in output_size must be positive integers.")

    # Generate the implicit uniform grid
    # base_grid has shape (D_out, H_out, W_out, 3)
    base_grid = _generate_grid(size, align_corners, device, dtype)

    # Expand grid for batch N. Use .expand() to avoid data copy.
    # Target shape for grid_for_mimic: (N, D_out, H_out, W_out, 3)
    grid_for_mimic = base_grid.unsqueeze(0).expand(N, -1, -1, -1, -1)

    return bi_to_tri(x, grid_for_mimic, align_corners, export_compatible)


def interpolate_3d_seq(
    x: torch.Tensor,
    new_size: Tuple[int, int, int],
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Resizes a 3D tensor using sequential bilinear interpolations.
    This is a separable interpolation approach, NOT true trilinear interpolation.
    Args:
        x: Input tensor of shape (N, C, D, H, W).
        new_size: Target size (new_D, new_H, new_W).
        align_corners: Passed to F.interpolate.
    Returns:
        Interpolated tensor of shape (N, C, new_D, new_H, new_W).
    """
    N, C, D, H, W = x.shape
    new_D, new_H, new_W = new_size

    # Step 1: Interpolate (H, W) using bilinear for each D-slice
    # Reshape so that each (D) slice is treated as a batch item for 2D interpolation
    x_hw_interp = x.reshape(N * C * D, 1, H, W)
    x_hw_interp = F.interpolate(
        x_hw_interp, size=(new_H, new_W), mode="bilinear", align_corners=align_corners
    )
    x_hw_interp = x_hw_interp.reshape(N, C, D, new_H, new_W)
    # Now shape is (N, C, D, new_H, new_W)

    # Step 2: Interpolate D to new_D.
    # We can do this by permuting D to be one of the last two dimensions for F.interpolate.
    # For example, interpolate along D and new_W (which is already at its target size).
    # This effectively becomes a 1D linear interpolation along the D-axis for each (N, C, new_H, w_idx) "vector".

    # Permute to (N, C, new_H, new_W, D) to make D the last dimension
    x_d_permuted = x_hw_interp.permute(0, 1, 3, 4, 2)
    # Reshape for F.interpolate (treating new_H*new_W as batch, D as length)
    # Each (n,c,h,w) element has a D-length vector to be interpolated.
    x_d_reshaped = x_d_permuted.reshape(
        N * C * new_H * new_W, 1, D
    )  # Shape for 1D interpolation (N_eff, Channels, Length)

    # Interpolate along the D dimension (now the last dimension of the reshaped tensor)
    # For 1D interpolation, F.interpolate expects 3D input (N, C, L) or 2D input (N,L)
    # mode 'linear' is for 1D data.
    x_d_interpolated = F.interpolate(
        x_d_reshaped, size=(new_D,), mode="linear", align_corners=align_corners
    )
    # Output shape: (N*C*new_H*new_W, 1, new_D)

    # Reshape back to desired intermediate order
    x_final_reshaped = x_d_interpolated.reshape(N, C, new_H, new_W, new_D)

    # Permute to final target shape (N, C, new_D, new_H, new_W)
    output = x_final_reshaped.permute(0, 1, 4, 2, 3)

    return output

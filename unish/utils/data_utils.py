# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from typing import Tuple
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a depth map to world coordinates (HxWx3) given the camera extrinsic and intrinsic.
    Returns both the world coordinates and the intermediate camera coordinates,
    as well as a mask for valid depth.

    Args:
        depth_map (np.ndarray):
            Depth map of shape (H, W).
        extrinsic (np.ndarray):
            Extrinsic matrix of shape (3, 4), representing the camera pose in OpenCV convention (w2c).
        intrinsic (np.ndarray):
            Intrinsic matrix of shape (3, 3).
        eps (float):
            Small epsilon for thresholding valid depth.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (world_coords_points, cam_coords_points, point_mask)

            - world_coords_points: (H, W, 3) array of 3D points in world frame.
            - cam_coords_points: (H, W, 3) array of 3D points in camera frame.
            - point_mask: (H, W) boolean array where True indicates valid (non-zero) depth.
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # The extrinsic is camera-from-world, so invert it to transform camera->world
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]
    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = (
        np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world
    ) # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(
    depth_map: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """
    Unprojects a depth map into camera coordinates, returning (H, W, 3).

    Args:
        depth_map (np.ndarray):
            Depth map of shape (H, W).
        intrinsic (np.ndarray):
            3x3 camera intrinsic matrix.
            Assumes zero skew and standard OpenCV layout:
            [ fx   0   cx ]
            [  0  fy   cy ]
            [  0   0    1 ]

    Returns:
        np.ndarray:
            An (H, W, 3) array, where each pixel is mapped to (x, y, z) in the camera frame.
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert (
        intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0
    ), "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def rotmat_to_aa(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3,3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    # Handle any remaining NaN or inf values
    aa = torch.where(torch.isfinite(aa), aa, torch.zeros_like(aa))
    return aa

def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    # Add numerical stability: ensure sin_squared_theta is non-negative
    eps = 1e-8
    sin_squared_theta = torch.clamp(sin_squared_theta, min=0.0)
    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta + eps)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    # Add numerical stability: avoid division by very small sin_theta values
    threshold = 1e-6
    sin_theta_safe = torch.where(sin_theta < threshold, threshold, sin_theta)
    k_pos: torch.Tensor = two_theta / sin_theta_safe
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    
    # Use a more conservative threshold for numerical stability
    k: torch.Tensor = torch.where(sin_squared_theta > threshold * threshold, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    
    # Add numerical stability to avoid division by zero or very small numbers
    denominator = torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                            t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    denominator = torch.clamp(denominator, min=eps)
    q /= denominator
    q *= 0.5
    
    # Normalize quaternion to unit length for additional stability
    q_norm = torch.norm(q, p=2, dim=-1, keepdim=True)
    q_norm = torch.clamp(q_norm, min=eps)
    q = q / q_norm
    
    return q

def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    assert len(x.shape) == 2 and x.shape[1] == 6
    x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous() # (B, 6) -> (B, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)

def rotmat_to_rot6d(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to 6D rotation representation.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,3,3) Batch of 3x3 rotation matrices.
    Returns:
        torch.Tensor: Batch of corresponding 6D rotation representations with shape (B,6).
    """
    assert len(x.shape) == 3 and x.shape[1] == 3 and x.shape[2] == 3
    # Output in column order: first 3 elements are the first column, next 3 elements are the second column
    # This matches the reshape operation in rot6d_to_rotmat
    return torch.cat([x[:, :, 0], x[:, :, 1]], dim=-1)

def aa_to_rot6d(x: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to 6D rotation representation.
    Args:
        x (torch.Tensor): (B,3) Batch of axis-angle representations.
    Returns:
        torch.Tensor: Batch of corresponding 6D rotation representations with shape (B,6).
    """
    assert len(x.shape) == 2 and x.shape[1] == 3
    x = aa_to_rotmat(x)
    x = rotmat_to_rot6d(x)
    return x

def rot6d_to_aa(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to axis-angle representation.
    Args:
        x (torch.Tensor): (B,6) Batch of 6D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding axis-angle representations with shape (B,3).
    """
    assert len(x.shape) == 2 and x.shape[1] == 6
    x = rot6d_to_rotmat(x)
    x = rotmat_to_aa(x)
    return x

def rotate_2d(pt_2d: np.array, rot_rad: float) -> np.array:
    """
    Rotate a 2D point on the x-y plane.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        rot_rad (float): Rotation angle
    Returns:
        np.array: Rotated 2D point.
    """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def convert_to_full_img_cam(pred_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
    s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
    tz = 2. * focal_length / (bbox_height * s)
    cx = 2. * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2. * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)
    trans = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return trans


def gen_trans_from_patch_cv(c_x: float, c_y: float,
                            src_width: float, src_height: float,
                            dst_width: float, dst_height: float,
                            scale: float, rot: float) -> np.array:
    """
    Create transformation matrix for the bounding box crop.
    Args:
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        src_width (float): Bounding box width.
        src_height (float): Bounding box height.
        dst_width (float): Output box width.
        dst_height (float): Output box height.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        trans (np.array): Target geometric transformation.
    """
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def generate_image_patch_cv2(img: np.array, c_x: float, c_y: float,
                             bb_width: float, bb_height: float,
                             patch_width: float, patch_height: float,
                             do_flip: bool, scale: float, rot: float,
                             border_mode=cv2.BORDER_CONSTANT, border_value=0) -> Tuple[np.array, np.array]:
    """
    Crop the input image and return the crop and the corresponding transformation matrix.
    Args:
        img (np.array): Input image of shape (H, W, 3)
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        bb_width (float): Bounding box width.
        bb_height (float): Bounding box height.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        do_flip (bool): Whether to flip image or not.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        img_patch (np.array): Cropped image patch of shape (patch_height, patch_height, 3)
        trans (np.array): Transformation matrix.
    """

    img_height, img_width, img_channels = img.shape
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1


    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), 
                        flags=cv2.INTER_LINEAR, 
                        borderMode=border_mode,
                        borderValue=border_value,
                )
    # Force borderValue=cv2.BORDER_CONSTANT for alpha channel
    if (img.shape[2] == 4) and (border_mode != cv2.BORDER_CONSTANT):
        img_patch[:,:,3] = cv2.warpAffine(img[:,:,3], trans, (int(patch_width), int(patch_height)), 
                                            flags=cv2.INTER_LINEAR, 
                                            borderMode=cv2.BORDER_CONSTANT,
                            )

    return img_patch, trans

def depth_to_pointmap(depth_map: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Convert depth map and camera intrinsics to 3D point map.
    
    Args:
        depth_map (torch.Tensor): Depth map tensor of shape (B, S, H, W, 1) or (B, S, H, W)
        intrinsics (torch.Tensor): Camera intrinsics tensor of shape (B, S, 3, 3)
        
    Returns:
        torch.Tensor: Point map tensor of shape (B, S, H, W, 3) where last dim is [X, Y, Z]
    """
    # Handle different input shapes for depth_map
    if depth_map.dim() == 5:  # (B, S, H, W, 1)
        depth = depth_map.squeeze(-1)  # (B, S, H, W)
    elif depth_map.dim() == 4:  # (B, S, H, W)
        depth = depth_map
    else:
        raise ValueError(f"Expected depth_map to have 4 or 5 dimensions, got {depth_map.dim()}")
    
    B, S, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype
    
    # Extract camera parameters from intrinsics
    # intrinsics shape: (B, S, 3, 3)
    fx = intrinsics[..., 0, 0]  # (B, S)
    fy = intrinsics[..., 1, 1]  # (B, S)  
    cx = intrinsics[..., 0, 2]  # (B, S)
    cy = intrinsics[..., 1, 2]  # (B, S)
    
    # Create coordinate grids
    # Generate u, v coordinates for all pixels
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing='ij'
    )
    
    # Expand to match batch dimensions: (1, 1, H, W)
    u_coords = u_coords.unsqueeze(0).unsqueeze(0).expand(B, S, H, W)
    v_coords = v_coords.unsqueeze(0).unsqueeze(0).expand(B, S, H, W)
    
    # Expand camera parameters to match spatial dimensions: (B, S, H, W)
    fx_expanded = fx.unsqueeze(-1).unsqueeze(-1).expand(B, S, H, W)
    fy_expanded = fy.unsqueeze(-1).unsqueeze(-1).expand(B, S, H, W)
    cx_expanded = cx.unsqueeze(-1).unsqueeze(-1).expand(B, S, H, W)
    cy_expanded = cy.unsqueeze(-1).unsqueeze(-1).expand(B, S, H, W)
    
    # Convert pixel coordinates to normalized camera coordinates
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy  
    # Z = depth
    X = (u_coords - cx_expanded) * depth / fx_expanded
    Y = (v_coords - cy_expanded) * depth / fy_expanded
    Z = depth
    
    # Stack to create point map: (B, S, H, W, 3)
    point_map = torch.stack([X, Y, Z], dim=-1)
    
    return point_map

def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

def image_uv(height: int, width: int, left: int = None, top: int = None, right: int = None, bottom: int = None, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """
    Get image space UV grid, ranging in [0, 1]. 

    >>> image_uv(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]

    Args:
        width (int): image width
        height (int): image height

    Returns:
        torch.Tensor: shape (height, width, 2)
    """
    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = torch.linspace((left + 0.5) / width, (right - 0.5) / width, right - left, device=device, dtype=dtype)
    v = torch.linspace((top + 0.5) / height, (bottom - 0.5) / height, bottom - top, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv

def unproject_cv(
    uv_coord: torch.Tensor,
    depth: torch.Tensor = None,
    extrinsics: torch.Tensor = None,
    intrinsics: torch.Tensor = None
) -> torch.Tensor:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    Args:
        uv_coord (torch.Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (torch.Tensor): [..., N] depth value
        extrinsics (torch.Tensor): [..., 4, 4] extrinsics matrix
        intrinsics (torch.Tensor): [..., 3, 3] intrinsics matrix

    Returns:
        points (torch.Tensor): [..., N, 3] 3d points
    """
    assert intrinsics is not None, "intrinsics matrix is required"
    points = torch.cat([uv_coord, torch.ones_like(uv_coord[..., :1])], dim=-1)
    points = points @ torch.inverse(intrinsics).transpose(-2, -1)
    if depth is not None:
        points = points * depth[..., None]
    if extrinsics is not None:
        points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        points = (points @ torch.inverse(extrinsics).transpose(-2, -1))[..., :3]
    return points

def depth_to_points(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor = None):
    height, width = depth.shape[-2:]
    uv = image_uv(width=width, height=height, dtype=depth.dtype, device=depth.device)
    pts = unproject_cv(uv, depth, intrinsics=intrinsics[..., None, :, :], extrinsics=extrinsics[..., None, :, :] if extrinsics is not None else None)
    return pts
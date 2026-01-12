import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import trimesh
import os

from unish.utils.renderer import Renderer
from unish.utils.data_utils import closed_form_inverse_se3, rotmat_to_aa, aa_to_rotmat


class SMPLWrapper:
    def __init__(self, model_folder='body_models/', model_type='smplx', device='cpu', dtype=torch.float32):
        """
        Initialize SMPL visualizer with SMPL or SMPL-X models
        
        Args:
            model_folder (str): Path to model folder (should contain smpl/ or smplx/ subfolders)
            model_type (str): Model type, either 'smpl' or 'smplx'
            device (str or torch.device): Device to run models on ('cpu', 'cuda', or torch.device object)
            dtype (torch.dtype): Data type for the models (default: torch.float32)
        """
        import smplx
        
        self.model_folder = model_folder
        self.model_type = model_type.lower()
        # 🚀 Ensure device is correctly converted to torch.device object
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.dtype = dtype
        self.models = {}
        
        # Initialize models for different genders on specified device
        if self.model_type == 'smplx':
            model_path = os.path.join(model_folder, 'smplx/models') if 'smplx' not in model_folder else model_folder
            self.models['male'] = smplx.create(model_path, model_type='smplx',
                                              gender='neutral',
                                              ext='npz',
                                              flat_hand_mean=True,
                                              num_betas=11,
                                              use_pca=False).to(self.device, dtype=self.dtype)
            
            self.models['female'] = smplx.create(model_path, model_type='smplx',
                                                gender='female',
                                                ext='npz',
                                                num_betas=11,
                                                flat_hand_mean=True,
                                                use_pca=False).to(self.device, dtype=self.dtype)
            
            self.models['neutral'] = smplx.create(model_path, model_type='smplx',
                                                 gender='neutral',
                                                 ext='npz',
                                                 flat_hand_mean=True,
                                                 num_betas=11,
                                                 use_pca=False).to(self.device, dtype=self.dtype)
        
        elif self.model_type == 'smpl':
            model_path = os.path.join(model_folder, 'smpl') if 'smpl' not in model_folder else model_folder
            self.models['male'] = smplx.create(model_path, model_type='smpl',
                                              gender='male',
                                              ext='pkl',
                                              num_betas=10).to(self.device, dtype=self.dtype)
            
            self.models['female'] = smplx.create(model_path, model_type='smpl',
                                                gender='female',
                                                ext='pkl',
                                                num_betas=10).to(self.device, dtype=self.dtype)
            
            self.models['neutral'] = smplx.create(model_path, model_type='smpl',
                                                 gender='neutral',
                                                 ext='pkl',
                                                 num_betas=10).to(self.device, dtype=self.dtype)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Please use 'smpl' or 'smplx'.")

    def get_vertices(self, poses, betas, trans, gender):
        """
        Get model vertices and joints for given parameters
        
        Args:
            poses (torch.Tensor): Pose parameters (72 for SMPL, 165 for SMPL-X)
            betas (torch.Tensor): Shape parameters (10 for SMPL, 11 for SMPL-X)
            trans (torch.Tensor): Translation parameters
            gender (str): Gender of the model ('male', 'female', or 'neutral')
            
        Returns:
            tuple: (vertices, joints)
        """
        if gender not in self.models:
            raise ValueError('Please provide gender as male, female, or neutral')
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)
        if isinstance(betas, np.ndarray):
            betas = torch.from_numpy(betas)
        if isinstance(trans, np.ndarray):
            trans = torch.from_numpy(trans)
        
        if len(poses.shape) == 1:
            poses = poses.unsqueeze(0)
        if len(betas.shape) == 1:
            betas = betas.unsqueeze(0)
        if len(trans.shape) == 1:
            trans = trans.unsqueeze(0)

        # Move data to the same device as the model
        poses = poses.to(self.device, dtype=self.dtype)
        betas = betas.to(self.device, dtype=self.dtype)
        trans = trans.to(self.device, dtype=self.dtype)
        
        if self.model_type == 'smplx':
            # SMPL-X parameters
            model_out = self.models[gender](
                betas=betas,
                global_orient=poses[:, :3],
                body_pose=poses[:, 3:66],
                left_hand_pose=poses[:, 75:120],
                right_hand_pose=poses[:, 120:165],
                jaw_pose=poses[:, 66:69],
                leye_pose=poses[:, 69:72],
                reye_pose=poses[:, 72:75],
                transl=trans
            )
        elif self.model_type == 'smpl':
            # SMPL parameters (72 dimensions: 3 for global_orient + 69 for body_pose)
            model_out = self.models[gender](
                betas=betas,
                global_orient=poses[:, :3],
                body_pose=poses[:, 3:72],
                transl=trans
            )

        return model_out.vertices[0], model_out.joints[0]

    def get_batch_vertices(self, poses, betas, trans, gender):
        """
        Get model vertices and joints for batched parameters (More efficient batch version)
        
        Args:
            poses (torch.Tensor): Pose parameters [B, 72/165] (72 for SMPL, 165 for SMPL-X)
            betas (torch.Tensor): Shape parameters [B, 10/11] (10 for SMPL, 11 for SMPL-X)
            trans (torch.Tensor): Translation parameters [B, 3]
            gender (str): Gender of the model ('male', 'female', or 'neutral')
            
        Returns:
            tuple: (vertices [B, V, 3], joints [B, J, 3])
        """

        assert len(poses.shape) == 2 and len(betas.shape) == 2 and len(trans.shape) == 2, "poses, betas, trans should be 2D"

        if gender not in self.models:
            raise ValueError('Please provide gender as male, female, or neutral')
        
        # Move data to the same device and dtype as the model
        poses = poses.to(self.device, dtype=self.dtype)
        betas = betas.to(self.device, dtype=self.dtype) 
        trans = trans.to(self.device, dtype=self.dtype)
        
        if self.model_type == 'smplx':
            # SMPL-X parameters
            model_out = self.models[gender](
                betas=betas,
                global_orient=poses[:, :3],
                body_pose=poses[:, 3:66],
                left_hand_pose=poses[:, 75:120],
                right_hand_pose=poses[:, 120:165],
                jaw_pose=poses[:, 66:69],
                leye_pose=poses[:, 69:72],
                reye_pose=poses[:, 72:75],
                transl=trans
            )
        elif self.model_type == 'smpl':
            # SMPL parameters (72 dimensions: 3 for global_orient + 69 for body_pose)
            model_out = self.models[gender](
                betas=betas,
                global_orient=poses[:, :3],
                body_pose=poses[:, 3:72],
                transl=trans
            )
                
        return model_out.vertices, model_out.joints

    def render(self, poses, betas, trans, gender, K, background=None, w2c=None):
        """
        Render SMPL model with given parameters
        
        Args:
            poses (torch.Tensor): Pose parameters (72 for SMPL, 165 for SMPL-X)
            betas (torch.Tensor): Shape parameters (10 for SMPL, 11 for SMPL-X)
            trans (torch.Tensor): Translation parameters
            gender (str): Gender of the model
            K (torch.Tensor): Camera intrinsic matrix
            background (numpy.ndarray, optional): Background image
            w2c (torch.Tensor, optional): Transformation matrix from world to camera
            
        Returns:
            tuple: (rendered_image, vertices)
        """
        
        extr = torch.eye(4) if w2c is None else w2c

        vertices, joints = self.get_vertices(poses, betas, trans, gender)

        if background is None:
            width, height = K[0, 2] * 2, K[1, 2] * 2
            background = np.zeros((int(height), int(width), 3))
        else:
            height, width = background.shape[:2]
            
        renderer = Renderer(width, height, device="cuda", faces=self.models[gender].faces, K=K)
        renderer.create_camera(R=extr[:3, :3], T=extr[:3, 3])

        vertices_float32 = vertices.float().to(self.device)
        render_img = renderer.render_mesh(vertices_float32, background, [0.8, 0.8, 0.8])
        return render_img, vertices
    
    def get_smpl_depth(self, vertices, K, extr=None, width=None, height=None, return_visible_vertices=False):
        """
        Get depth map and mask from SMPL vertices
        
        Args:
            vertices (torch.Tensor): SMPL vertices [V, 3]
            K (torch.Tensor): Camera intrinsic matrix [3, 3]
            extr (torch.Tensor, optional): Camera extrinsic matrix [4, 4]. If None, uses identity
            width (int, optional): Image width. If None, uses 2*K[0,2]
            height (int, optional): Image height. If None, uses 2*K[1,2]
            
        Returns:
            tuple: (depth_map, mask)
                - depth_map (np.ndarray): Depth map as numpy array
                - mask (np.ndarray): Object mask as numpy array
        """
        # Set default extrinsic matrix if not provided
        if extr is None:
            extr = torch.eye(4, device=vertices.device, dtype=vertices.dtype)

        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        
        # Get image dimensions from camera intrinsics if not provided
        if width is None:
            width = int(K[0, 2] * 2)
        if height is None:
            height = int(K[1, 2] * 2)
        
        # Ensure vertices are on the correct device
        vertices = vertices.to(self.device, dtype=self.dtype)
        K = K.to(self.device, dtype=self.dtype)
        extr = extr.to(self.device, dtype=self.dtype)
        
        # Create renderer instance
        renderer = Renderer(width, height, device=self.device, faces=self.models['neutral'].faces, K=K)
        
        # Set camera pose from extrinsic matrix
        R = extr[:3, :3]  # Rotation matrix
        T = extr[:3, 3]   # Translation vector
        renderer.create_camera(R=R, T=T)
        
        # Render depth only
        if return_visible_vertices:
            depth_map, mask, visible_vertices = renderer.render_depth_only(vertices, return_visible_vertices=return_visible_vertices)
            return depth_map, mask, visible_vertices
        else:
            depth_map, mask = renderer.render_depth_only(vertices, return_visible_vertices=return_visible_vertices)
            return depth_map, mask
        
    def get_smplx_vertices(self, poses, betas, trans, gender):
        """Deprecated: Use get_vertices() instead. This method is kept for backward compatibility."""
        return self.get_vertices(poses, betas, trans, gender)
    
    def get_smplx_batch_vertices(self, poses, betas, trans, gender):
        """Deprecated: Use get_batch_vertices() instead. This method is kept for backward compatibility."""
        return self.get_batch_vertices(poses, betas, trans, gender)

def transform_smpl(smpl_dict, extrinsics, copy_dict=True):
    """
    Transform SMPL parameters from camera coordinate system to world coordinate system.
    
    Args:
        smpl_dict (dict): Dictionary containing SMPL parameters in camera coordinates
            - 'pose_cam': Pose parameters as rotation matrices (B, S, N, 3, 3) or axis-angle (B, S, N*3)
            - 'trans_cam': Translation parameters (B, S, 3) 
            - 'betas': Shape parameters (B, S, 10 or 11) - unchanged by coordinate transform
        extrinsics (torch.Tensor): Camera extrinsic matrix (B, S, 4, 4) or (4, 4)
            Transformation matrix from world to camera coordinates
        copy_dict (bool): Whether to create a copy of the input dict (default: True)
    
    Returns:
        dict: Transformed SMPL dictionary with parameters in world coordinates
            - 'pose_world': Transformed pose parameters in world coordinates
            - 'trans_world': Transformed translation parameters in world coordinates
            - 'betas': Shape parameters (unchanged)
    """
    
    # Create a copy to avoid modifying the original dictionary
    if copy_dict:
        transformed_dict = {}
        for key, value in smpl_dict.items():
            if torch.is_tensor(value):
                transformed_dict[key] = value.clone()
            else:
                transformed_dict[key] = value
    else:
        transformed_dict = smpl_dict
    
    # Get batch and sequence dimensions from camera coordinate parameters
    pose_cam = smpl_dict['pose_cam']
    trans_cam = smpl_dict['trans_cam']
    
    batch_size = pose_cam.shape[0]
    seq_len = pose_cam.shape[1]
    
    # Handle extrinsics shape - ensure it has batch and sequence dimensions
    if len(extrinsics.shape) == 2:  # (4, 4)
        extrinsics = extrinsics.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
    elif len(extrinsics.shape) == 3:  # (B, 4, 4)
        extrinsics = extrinsics.unsqueeze(1).expand(-1, seq_len, -1, -1)
    elif len(extrinsics.shape) == 4:  # (B, S, 4, 4)
        pass  # Already correct shape
    else:
        raise ValueError(f"Unsupported extrinsics shape: {extrinsics.shape}")
    
    # Use closed-form inverse for SE3 matrices instead of torch.inverse
    extrinsics_flat = extrinsics.view(-1, 4, 4)
    extrinsics_inv_flat = closed_form_inverse_se3(extrinsics_flat)
    cam_to_world_extrinsics = extrinsics_inv_flat.view(batch_size, seq_len, 4, 4)
    
    # Extract rotation and translation from camera-to-world extrinsics
    cam_to_world_R = cam_to_world_extrinsics[:, :, :3, :3]  # (B, S, 3, 3)
    cam_to_world_t = cam_to_world_extrinsics[:, :, :3, 3]   # (B, S, 3)
    
    # Transform translation from camera space to world space
    # world_trans = R * cam_trans + t
    trans_cam_flat = trans_cam.view(batch_size * seq_len, 3, 1)  # (B*S, 3, 1)
    cam_to_world_R_flat = cam_to_world_R.view(batch_size * seq_len, 3, 3)  # (B*S, 3, 3)
    cam_to_world_t_flat = cam_to_world_t.view(batch_size * seq_len, 3, 1)  # (B*S, 3, 1)
    
    # Apply rotation and translation
    transformed_trans = torch.bmm(cam_to_world_R_flat, trans_cam_flat) + cam_to_world_t_flat
    transformed_dict['trans_world'] = transformed_trans.view(batch_size, seq_len, 3)
    
    # Transform pose parameters
    # Only transform the root joint (global_orient), keep body_pose unchanged
    if len(pose_cam.shape) == 5 and pose_cam.shape[-2:] == (3, 3):
        # Rotation matrix format (B, S, N, 3, 3)
        pose_world = pose_cam.clone()
        
        # Only transform the first joint (root/global_orient) using similarity transformation
        root_joint_rot = pose_cam[:, :, 0]  # (B, S, 3, 3)
        root_joint_flat = root_joint_rot.view(batch_size * seq_len, 3, 3)  # (B*S, 3, 3)
        cam_to_world_R_T = cam_to_world_R_flat.transpose(-2, -1)  # (B*S, 3, 3)
        
        # Apply coordinate transformation: R_world = R_cam_to_world @ R_cam
        # For rotation matrices, coordinate transformation is direct multiplication
        transformed_root_rot = torch.bmm(cam_to_world_R_flat, root_joint_flat)  # R_cam_to_world @ R_cam
        
        # Replace only the root joint, keep all other joints unchanged
        pose_world[:, :, 0] = transformed_root_rot.view(batch_size, seq_len, 3, 3)
        transformed_dict['pose_world'] = pose_world
        
    elif len(pose_cam.shape) == 3:
        # Axis-angle format (B, S, N*3) - typically 72 for SMPL or 165 for SMPL-X
        pose_world = pose_cam.clone()
        
        # Only transform the first 3 parameters (root joint / global_orient)
        root_joint_aa = pose_cam[:, :, :3]  # (B, S, 3)
        root_joint_flat = root_joint_aa.view(batch_size * seq_len, 3)  # (B*S, 3)
        
        # Convert root joint from axis-angle to rotation matrix
        root_joint_rotmat = aa_to_rotmat(root_joint_flat)  # (B*S, 3, 3)
        cam_to_world_R_T = cam_to_world_R_flat.transpose(-2, -1)  # (B*S, 3, 3)
        
        # Apply coordinate transformation: R_world = R_cam_to_world @ R_cam
        # For rotation matrices, coordinate transformation is direct multiplication
        transformed_root_rotmat = torch.bmm(cam_to_world_R_flat, root_joint_rotmat)  # R_cam_to_world @ R_cam
        
        # Convert back to axis-angle
        transformed_root_aa = rotmat_to_aa(transformed_root_rotmat)  # (B*S, 3)
        
        # Replace only the first 3 parameters (root joint), keep all others unchanged
        pose_world[:, :, :3] = transformed_root_aa.view(batch_size, seq_len, 3)
        transformed_dict['pose_world'] = pose_world
        
    else:
        raise ValueError(f"Unsupported pose format with shape: {pose_cam.shape}")
    
    # Shape parameters (betas) remain unchanged as they are not affected by coordinate transformations
    # No need to transform betas
    
    return transformed_dict

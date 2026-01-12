import torch
import numpy as np

from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer.cameras import look_at_rotation
from pytorch3d.transforms import axis_angle_to_matrix


colors_str_map = {
    "gray": [0.8, 0.8, 0.8],
    "green": [39, 194, 128],
}


def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    
    # Ensure bbox coordinates are within valid range
    h_bg, w_bg = background.shape[:2]
    left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Limit bbox coordinates within background image range
    left = max(0, min(left, w_bg))
    top = max(0, min(top, h_bg))
    right = max(left + 1, min(right, w_bg))  # Ensure right > left
    bottom = max(top + 1, min(bottom, h_bg))  # Ensure bottom > top
    
    roi_image = out_image[top:bottom, left:right]
    
    # Check if roi_image is empty
    if roi_image.size == 0:
        print(f"Warning: ROI image is empty. bbox: [{left}, {top}, {right}, {bottom}], bg_shape: {background.shape}")
        return out_image
    
    # Check if dimensions of image and mask match roi_image
    expected_height, expected_width = bottom - top, right - left
    
    if image.shape[:2] != (expected_height, expected_width):
        print(f"Warning: Image shape mismatch. Expected: ({expected_height}, {expected_width}), Got: {image.shape[:2]}")
        # If dimensions mismatch, use background color
        return out_image
        
    if mask.shape[:2] != (expected_height, expected_width):
        print(f"Warning: Mask shape mismatch. Expected: ({expected_height}, {expected_width}), Got: {mask.shape[:2]}")
        # If dimensions mismatch, use background color
        return out_image
    
    # Safely apply mask
    try:
        roi_image[mask] = image[mask]
        out_image[top:bottom, left:right] = roi_image
    except Exception as e:
        print(f"Error in overlay operation: {e}")
        print(f"roi_image shape: {roi_image.shape}, mask shape: {mask.shape}, image shape: {image.shape}")
        # If error occurs, return original background
        return out_image

    return out_image

def overlay_depth_onto_background(depth_map, mask, bbox, background_shape, background_value=0.0):
    """
    Overlay depth map onto full-size background with differentiable torch operations
    
    Args:
        depth_map: torch.Tensor, depth map from rasterizer
        mask: torch.Tensor, object mask (boolean or float)
        bbox: torch.Tensor, bounding box coordinates [left, top, right, bottom]
        background_shape: tuple, (height, width) of output
        background_value: float, background depth value
        
    Returns:
        torch.Tensor, full-size depth map
        torch.Tensor, full-size mask
    """
    device = depth_map.device
    dtype = depth_map.dtype
    
    # Ensure inputs are torch tensors
    if not isinstance(depth_map, torch.Tensor):
        depth_map = torch.tensor(depth_map, device=device, dtype=dtype)
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, device=device, dtype=torch.float32)
    
    h_bg, w_bg = background_shape
    
    # Create full-size background tensors
    out_depth = torch.full((h_bg, w_bg), background_value, device=device, dtype=dtype)
    out_mask = torch.zeros((h_bg, w_bg), device=device, dtype=mask.dtype)
    
    # Extract bbox coordinates and clamp to valid range
    if bbox.dim() > 1:
        bbox = bbox[0]  # Take first bbox if batched
    
    bbox = bbox.to(torch.int32)
    left = torch.clamp(bbox[0], 0, w_bg)
    top = torch.clamp(bbox[1], 0, h_bg)
    right = torch.clamp(bbox[2], left + 1, w_bg)
    bottom = torch.clamp(bbox[3], top + 1, h_bg)
    
    # Calculate expected dimensions
    expected_height = bottom - top
    expected_width = right - left
    
    # Check if dimensions match
    if (depth_map.shape[0] != expected_height or depth_map.shape[1] != expected_width):
        print(f"Warning: Depth map shape mismatch. Expected: ({expected_height}, {expected_width}), Got: {depth_map.shape[:2]}")
        return out_depth, out_mask
        
    if (mask.shape[0] != expected_height or mask.shape[1] != expected_width):
        print(f"Warning: Mask shape mismatch. Expected: ({expected_height}, {expected_width}), Got: {mask.shape[:2]}")
        return out_depth, out_mask
    
    # Overlay depth map and mask onto background using differentiable operations
    try:
        # Convert coordinates to integers for indexing (non-differentiable but necessary)
        left_int = left.item()
        top_int = top.item()
        right_int = right.item()
        bottom_int = bottom.item()
        
        # Use tensor operations for differentiable overlay
        out_depth[top_int:bottom_int, left_int:right_int] = depth_map
        out_mask[top_int:bottom_int, left_int:right_int] = mask
        
    except Exception as e:
        print(f"Error in depth overlay operation: {e}")
        print(f"Target region shape: ({bottom_int - top_int}, {right_int - left_int}), depth_map shape: {depth_map.shape}")
        
    return out_depth, out_mask


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype

    K = torch.zeros((K_org.shape[0], 4, 4)).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1

    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    z_coords = x3d[..., 2:]
    
    valid_z_mask = (torch.abs(z_coords) >= 1e-5) & torch.isfinite(z_coords)
    
    z_coords_safe = torch.where(valid_z_mask, z_coords, torch.sign(z_coords) * 1e-5 + 1e-5)
    
    x3d_safe = x3d.clone()
    x3d_safe[..., 2:] = z_coords_safe
    x2d = torch.div(x3d_safe, x3d_safe[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    
    final_valid_mask = valid_z_mask.squeeze(-1) & torch.isfinite(x2d).all(dim=-1, keepdim=True).squeeze(-1)
    
    x2d_masked = torch.where(final_valid_mask.unsqueeze(-1), x2d, torch.tensor(-999.0, device=x2d.device, dtype=x2d.dtype))
    
    return x2d_masked


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    if X.numel() == 0:
        print("Warning: Empty points for bbox computation, using full image bbox")
        bbox = torch.tensor([[0, 0, img_w, img_h]]).float()
        return bbox
    
    if len(X.shape) == 3:
        X_flat = X.reshape(-1, X.shape[-1])  # (batch_size * num_points, 2)
    elif len(X.shape) == 2:
        X_flat = X
    else:
        print(f"Warning: Unexpected X shape {X.shape}, using full image bbox")
        bbox = torch.tensor([[0, 0, img_w, img_h]]).float()
        return bbox
    
    valid_mask = torch.isfinite(X_flat).all(dim=-1)
    if not valid_mask.any():
        print("Warning: No valid points for bbox computation, using full image bbox")
        bbox = torch.tensor([[0, 0, img_w, img_h]]).float()
        return bbox
    
    X_valid = X_flat[valid_mask]
    if X_valid.numel() == 0:
        print("Warning: No valid points after filtering, using full image bbox")
        bbox = torch.tensor([[0, 0, img_w, img_h]]).float()
        return bbox
    
    img_w_tensor = torch.tensor(img_w, dtype=X_valid.dtype, device=X_valid.device)
    img_h_tensor = torch.tensor(img_h, dtype=X_valid.dtype, device=X_valid.device)
    
    left = torch.clamp(X_valid[:, 0].min(), min=0, max=img_w_tensor)
    right = torch.clamp(X_valid[:, 0].max(), min=0, max=img_w_tensor)
    top = torch.clamp(X_valid[:, 1].min(), min=0, max=img_h_tensor)
    bottom = torch.clamp(X_valid[:, 1].max(), min=0, max=img_h_tensor)
    
    if left >= right:
        left = torch.clamp(left - 10, min=0, max=img_w_tensor)
        right = torch.clamp(left + 20, min=1, max=img_w_tensor)
    if top >= bottom:
        top = torch.clamp(top - 10, min=0, max=img_h_tensor)
        bottom = torch.clamp(top + 20, min=1, max=img_h_tensor)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = right - left
    height = bottom - top

    img_w_tensor = torch.tensor(img_w, dtype=cx.dtype, device=cx.device)
    img_h_tensor = torch.tensor(img_h, dtype=cy.dtype, device=cy.device)
    scaleFactor_tensor = torch.tensor(scaleFactor, dtype=cx.dtype, device=cx.device)

    new_left = torch.clamp(cx - width / 2 * scaleFactor_tensor, min=0, max=img_w_tensor - 1)
    new_right = torch.clamp(cx + width / 2 * scaleFactor_tensor, min=1, max=img_w_tensor)
    new_top = torch.clamp(cy - height / 2 * scaleFactor_tensor, min=0, max=img_h_tensor - 1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor_tensor, min=1, max=img_h_tensor)
    
    if new_left >= new_right:
        new_left = torch.tensor(0, dtype=new_left.dtype, device=new_left.device)
        new_right = torch.tensor(max(1, min(img_w, 100)), dtype=new_right.dtype, device=new_right.device)
    if new_top >= new_bottom:
        new_top = torch.tensor(0, dtype=new_top.dtype, device=new_top.device)
        new_bottom = torch.tensor(max(1, min(img_h, 100)), dtype=new_bottom.dtype, device=new_bottom.device)

    bbox = torch.stack((new_left.detach(), new_top.detach(), new_right.detach(), new_bottom.detach())).int().float()
    
    if bbox.dim() == 1:
        bbox = bbox.unsqueeze(0)
    
    return bbox


class Renderer:
    def __init__(self, width, height, focal_length=None, device="cuda", faces=None, K=None, bin_size=0):
        """set bin_size to 0 for no binning"""
        self.width = width
        self.height = height
        self.bin_size = bin_size
        assert (focal_length is not None) ^ (K is not None), "focal_length and K are mutually exclusive"

        self.device = device
        if faces is not None:
            if isinstance(faces, np.ndarray):
                faces = torch.from_numpy((faces).astype("int"))
            self.faces = faces.unsqueeze(0).to(self.device)

        self.initialize_camera_params(focal_length, K)
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer()

    def create_renderer(self):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes[0], blur_radius=1e-5, bin_size=self.bin_size
                ),
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=self.lights,
            ),
        )

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        return PerspectiveCameras(
            device=self.device, R=self.R.mT, T=self.T, K=self.K_full, image_size=self.image_sizes, in_ndc=False
        )

    def initialize_camera_params(self, focal_length, K):
        # Extrinsics
        self.R = torch.diag(torch.tensor([1, 1, 1])).float().to(self.device).unsqueeze(0)

        self.T = torch.tensor([0, 0, 0]).unsqueeze(0).float().to(self.device)

        # Intrinsics
        if K is not None:
            self.K = K.float().reshape(1, 3, 3).to(self.device)
        else:
            assert focal_length is not None, "focal_length or K should be provided"
            self.K = (
                torch.tensor([[focal_length, 0, self.width / 2], [0, focal_length, self.height / 2], [0, 0, 1]])
                .float()
                .reshape(1, 3, 3)
                .to(self.device)
            )
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()

    def set_intrinsic(self, K):
        self.K = K.reshape(1, 3, 3)

    def update_bbox(self, x3d, scale=2.0, mask=None):
        """Update bbox of cameras from the given 3d points

        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """

        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1))

        if mask is not None:
            x2d = x2d[:, ~mask]

        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(
        self,
    ):
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def render_mesh(self, vertices, background=None, colors=None, VI=50):
        if colors is None:
            colors = [0.8, 0.8, 0.8]
        self.update_bbox(vertices[::VI], scale=1.2)
        vertices = vertices.unsqueeze(0)

        if isinstance(colors, torch.Tensor):
            # per-vertex color
            verts_features = colors.to(device=vertices.device, dtype=vertices.dtype)
            colors = [0.8, 0.8, 0.8]
        else:
            if colors[0] > 1:
                colors = [c / 255.0 for c in colors]
            verts_features = torch.tensor(colors).reshape(1, 1, 3).to(device=vertices.device, dtype=vertices.dtype)
            verts_features = verts_features.repeat(1, vertices.shape[1], 1)
        textures = TexturesVertex(verts_features=verts_features)

        mesh = Meshes(
            verts=vertices,
            faces=self.faces,
            textures=textures,
        )

        materials = Materials(device=self.device, specular_color=(colors,), shininess=0)

        results = torch.flip(self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights), [1, 2])
        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3

        if background is None:
            background = np.ones((self.height, self.width, 3)).astype(np.uint8) * 255

        image = overlay_image_onto_background(image, mask, self.bboxes, background.copy())
        self.reset_bbox()
        return image

    def render_with_ground(self, verts, colors, cameras, lights, faces=None):
        """
        :param verts (N, V, 3), potential multiple people
        :param colors (N, 3) or (N, V, 3)
        :param faces (N, F, 3), optional, otherwise self.faces is used will be used
        """
        # Sanity check of input verts, colors and faces: (B, V, 3), (B, F, 3), (B, V, 3)
        N, V, _ = verts.shape
        if faces is None:
            faces = self.faces.clone().expand(N, -1, -1)
        else:
            assert len(faces.shape) == 3, "faces should have shape of (N, F, 3)"

        assert len(colors.shape) in [2, 3]
        if len(colors.shape) == 2:
            assert len(colors) == N, "colors of shape 2 should be (N, 3)"
            colors = colors[:, None]
        colors = colors.expand(N, V, -1)[..., :3]

        # (V, 3), (F, 3), (V, 3)
        gv, gf, gc = self.ground_geometry
        verts = list(torch.unbind(verts, dim=0)) + [gv]
        faces = list(torch.unbind(faces, dim=0)) + [gf]
        colors = list(torch.unbind(colors, dim=0)) + [gc[..., :3]]
        mesh = create_meshes(verts, faces, colors)

        materials = Materials(device=self.device, shininess=0)

        results = self.renderer(mesh, cameras=cameras, lights=lights, materials=materials)
        image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

        return image
    
    def render_depth_only(self, vertices, VI=50, return_visible_vertices=False):
        """
        Render only the depth map without RGB computation.
        
        Args:
            vertices: Mesh vertices
            VI: Vertex interval for bbox computation
            return_visible_vertices: If True, also return visible vertex coordinates
            
        Returns:
            tuple: (depth_map, mask) or (depth_map, mask, visible_vertices)
                - depth_map: Depth map as numpy array, shape (height, width)
                - mask: Object mask as numpy array, shape (height, width)
                - visible_vertices: PyTorch tensor of visible vertex coordinates [N, 3]
        """
        self.update_bbox(vertices[::VI], scale=1.2)
        vertices = vertices.unsqueeze(0)

        # Create a simple mesh for depth rendering
        verts_features = torch.ones(1, vertices.shape[1], 3, device=vertices.device, dtype=vertices.dtype)
        textures = TexturesVertex(verts_features=verts_features)

        mesh = Meshes(
            verts=vertices,
            faces=self.faces,
            textures=textures,
        )

        # Get rasterizer fragments for depth information
        fragments = self.renderer.rasterizer(mesh, cameras=self.cameras)
        
        # Extract depth map from fragments
        depth_map = torch.flip(fragments.zbuf[0, ..., 0], [0, 1])  # Flip to match image orientation
        
        # Create mask from valid depth values
        mask = torch.flip(fragments.pix_to_face[0, ..., 0] >= 0, [0, 1])  # Valid faces have non-negative indices
        
        # Handle invalid depth values
        depth_map = torch.where(depth_map < 0, torch.tensor(0.0, device=depth_map.device), depth_map)
        
        visible_vertices = None
        if return_visible_vertices:
            # Extract visible vertices from fragments
            visible_vertices = self._get_visible_vertices(fragments, mesh, return_coords=True)
        
        # Apply overlay processing to ensure consistent size with other render methods
        depth_map_full, mask_full = overlay_depth_onto_background(
            depth_map, mask, self.bboxes, (self.height, self.width), background_value=0.0
        )
        
        self.reset_bbox()
        
        if return_visible_vertices:
            return depth_map_full, mask_full, visible_vertices
        else:
            return depth_map_full, mask_full

    def _get_visible_vertices(self, fragments, mesh, return_coords=False):
        """
        Get visible vertex indices from rasterizer fragments.
        
        Args:
            fragments: Rasterizer fragments containing pix_to_face
            mesh: The mesh object used for rendering
            return_coords: If True, return vertex coordinates instead of indices
            
        Returns:
            torch.Tensor: Tensor of unique visible vertex indices or coordinates
        """
        # Get the face indices for each pixel
        pix_to_face = fragments.pix_to_face[0]  # Remove batch dimension
        
        # Find pixels that have valid faces (face_id >= 0)
        valid_mask = pix_to_face >= 0
        
        # Get faces tensor
        faces = mesh.faces_list()[0]  # Get faces tensor (shape: [num_faces, 3])
        
        # Only consider the first (closest) face for each pixel
        closest_faces = pix_to_face[..., 0]  # [height, width]
        valid_closest = valid_mask[..., 0]  # [height, width]
        
        # Get unique visible face IDs
        visible_face_ids = closest_faces[valid_closest].unique()
        
        # Get all visible vertex IDs from visible faces
        # Use indexing to maintain differentiability
        visible_face_vertices = faces[visible_face_ids]  # [num_visible_faces, 3]
        visible_vertex_indices = visible_face_vertices.flatten().unique()  # Flatten and get unique vertices
        
        if return_coords:
            # Return actual vertex coordinates
            vertices = mesh.verts_list()[0]  # [num_vertices, 3]
            visible_vertex_coords = vertices[visible_vertex_indices]  # [num_visible_vertices, 3]
            return visible_vertex_coords
        else:
            # Return vertex indices
            return visible_vertex_indices


def create_meshes(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (B, F, 3)
    :param colors (B, V, 3)
    """
    textures = TexturesVertex(verts_features=colors)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return join_meshes_as_scene(meshes)


def get_global_cameras(verts, device="cuda", distance=5, position=(-5.0, 5.0, 0.0)):
    """This always put object at the center of view"""
    positions = torch.tensor([position]).repeat(len(verts), 1)
    targets = verts.mean(1)

    directions = targets - positions
    directions = directions / torch.norm(directions, dim=-1).unsqueeze(-1) * distance
    positions = targets - directions

    rotation = look_at_rotation(positions, targets).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)

    lights = PointLights(device=device, location=[position])
    return rotation, translation, lights


def get_global_cameras_static(
    verts, beta=4.0, cam_height_degree=30, target_center_height=1.0, use_long_axis=False, vec_rot=45, device="cuda"
):
    L, V, _ = verts.shape

    # Compute target trajectory, denote as center + scale
    targets = verts.mean(1)  # (L, 3)
    targets[:, 1] = 0  # project to xz-plane
    target_center = targets.mean(0)  # (3,)
    target_scale, target_idx = torch.norm(targets - target_center, dim=-1).max(0)

    # a 45 degree vec from longest axis
    if use_long_axis:
        long_vec = targets[target_idx] - target_center  # (x, 0, z)
        long_vec = long_vec / torch.norm(long_vec)
        R = axis_angle_to_matrix(torch.tensor([0, np.pi / 4, 0])).to(long_vec)
        vec = R @ long_vec
    else:
        vec_rad = vec_rot / 180 * np.pi
        vec = torch.tensor([np.sin(vec_rad), 0, np.cos(vec_rad)]).float()
        vec = vec / torch.norm(vec)

    # Compute camera position (center + scale * vec * beta) + y=4
    target_scale = max(target_scale, 1.0) * beta
    position = target_center + vec * target_scale
    position[1] = target_scale * np.tan(np.pi * cam_height_degree / 180) + target_center_height

    # Compute camera rotation and translation
    positions = position.unsqueeze(0).repeat(L, 1)
    target_centers = target_center.unsqueeze(0).repeat(L, 1)
    target_centers[:, 1] = target_center_height
    rotation = look_at_rotation(positions, target_centers).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)

    lights = PointLights(device=device, location=[position.tolist()])
    return rotation, translation, lights


def get_ground_params_from_points(root_points, vert_points):
    """xz-plane is the ground plane
    Args:
        root_points: (L, 3), to decide center
        vert_points: (L, V, 3), to decide scale
    """
    root_max = root_points.max(0)[0]  # (3,)
    root_min = root_points.min(0)[0]  # (3,)
    cx, _, cz = (root_max + root_min) / 2.0

    vert_max = vert_points.reshape(-1, 3).max(0)[0]  # (L, 3)
    vert_min = vert_points.reshape(-1, 3).min(0)[0]  # (L, 3)
    scale = (vert_max - vert_min)[[0, 2]].max()
    return float(scale), float(cx), float(cz)

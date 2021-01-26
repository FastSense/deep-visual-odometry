'''
3D geometry functions.
Vector and matrix transformations.
Point cloud utils.
Reprojection functions.
'''

from load_config import *

@tf.function
def _rot_mat_x(theta):
    '''
    theta: tensor of shape [batch_size]
    '''
    one = tf.ones_like(theta)
    zero = tf.zeros_like(theta)
    cos = tf.cos(theta)
    sin = tf.sin(theta)
    rotation_matrix = tf.stack([
            one, zero, zero,
            zero, cos, -sin,
            zero, sin, cos,
        ], axis=1)
    rotation_matrix = tf.reshape(rotation_matrix, (-1, 3, 3))
    return rotation_matrix

@tf.function
def _rot_mat_y(theta):
    '''
    theta: tensor of shape [batch_size]
    '''
    one = tf.ones_like(theta)
    zero = tf.zeros_like(theta)
    cos = tf.cos(theta)
    sin = tf.sin(theta)
    rotation_matrix = tf.stack([
            cos, zero, sin,
            zero, one, zero,
            -sin, zero, cos,
        ], axis=1)
    rotation_matrix = tf.reshape(rotation_matrix, (-1, 3, 3))
    return rotation_matrix

@tf.function
def _rot_mat_z(theta):
    '''
    theta: tensor of shape [batch_size]
    '''
    one = tf.ones_like(theta)
    zero = tf.zeros_like(theta)
    cos = tf.cos(theta)
    sin = tf.sin(theta)
    rotation_matrix = tf.stack([
            cos, -sin, zero,
            sin, cos, zero,
            zero, zero, one,
        ], axis=1)
    rotation_matrix = tf.reshape(rotation_matrix, (-1, 3, 3))
    return rotation_matrix

@tf.function
def rotation_matrix_from_euler(euler):
    '''
    euler: tensor of shape [batch_size, 3] -- euler angles
    returns: tensor of shape [batch_size, 3, 3] -- rotation matrix
    '''
    alpha = euler[:, 0]
    beta = euler[:, 1]
    gamma = euler[:, 2]
    rotation_matrix = _rot_mat_z(gamma) @ _rot_mat_y(beta) @ _rot_mat_x(alpha)
    return rotation_matrix

@tf.function
def rt_matrix_from_6_dof(prediction):
    '''
    prediciton: tensor of shape [batch_size, 6] -- 3 euler angles and 3-coordinate shift
    returns: tensor of shape [batch_size, 4, 4] -- rotation-translation matrix
    '''
    # Get rotation and translation
    rotation = prediction[:,:3]
    translation = prediction[:,3:]
    # Make rotation matrix
    rotation = rotation_matrix_from_euler(rotation) # [batch, 3, 3]
    # Make rotation-translation matrix
    rt_matrix = tf.concat([rotation, translation[:, :, None]], axis=2) # [batch, 3, 4]
    newline = tf.concat([
            tf.zeros_like(rt_matrix)[:, :1, :3],
            tf.ones_like(rt_matrix)[:, :1, :1]
            ], axis=2) # [batch, 1, 4]
    rt_matrix = tf.concat([rt_matrix, newline], axis=1) # [batch, 4, 4]
    return rt_matrix


@tf.function
def depth_to_point_cloud(depth, inv_camera_matrix, return_grid=False):
    # Matrix in image coordinates
    X, Y = np.meshgrid(
        (np.arange(0, config.image_w) + 0.5) / config.image_w * config.original_image_w - 0.5,
        (np.arange(0, config.image_h) + 0.5) / config.image_h * config.original_image_h - 0.5,
        # np.arange(0, config.image_w) / config.image_w * config.original_image_w,
        # np.arange(0, config.image_h) / config.image_h * config.original_image_h,
    )
    X = tf.constant(np.tile(X.reshape(1, 1, -1), [depth.shape[0], 1, 1]), dtype=depth.dtype)
    Y = tf.constant(np.tile(Y.reshape(1, 1, -1), [depth.shape[0], 1, 1]), dtype=depth.dtype)
    Z = tf.reshape(depth, [depth.shape[0], 1, -1])
    # Pixel coordinates
    pix_coords = tf.concat([X, Y, tf.ones_like(X)], 1) # [batch, 3, h*w]
    # Pixel coordinates to image coordinates
    img_coords = inv_camera_matrix[:3, :3] @ pix_coords # [batch, 3, h*w]
    # Image coordinates to camera coordinates
    cam_coords = tf.concat([
            img_coords * Z,
            tf.ones_like(X)], 1) # [batch, 4, h*w]
    
    if return_grid:
        return cam_coords, X, Y, Z
    
    return cam_coords

@tf.function
def get_warp(depth, transform, camera_matrix, inv_camera_matrix):
    '''
    depth:
        tf.Tensor of shape [batch_size, h, w{, 1}]
        Depth map of the ORIGINAL image (e.g. L - left camera)
    
    transform:
        tf.Tensor of shape [4, 4]
        Transformation matrix between the two camera coordinate systems
        (e.g. from L to R)
    
    Returns:
    warp:
        tf.Tensor of shape [batch_size, h, w, 2]
        Warping, which can shift image from original to the other view
    cam_coords:
        tf.Tensor of shape [batch, 4, h*w]
        Point cloud in the new coordinate system
    '''
    cam_coords, X, Y, Z = depth_to_point_cloud(depth, inv_camera_matrix, True) # [batch, 4, h*w]
    
    # Move to the coordinates of the second camera
    cam_coords = (camera_matrix @ transform) @ cam_coords # [batch, 4, h*w]
    
    # Camera coordinates to image coordinates (second camera)
    pix_coords_xy = cam_coords[:, :2] / cam_coords[:, 2:3] # [batch, 2, h*w]
    
    # Calculate shift between images
    pix_coords_orig = tf.concat([X, Y], 1) # [batch, 2, h*w]
    warp = pix_coords_xy - pix_coords_orig
    # For using with resized image
    warp = warp * tf.constant([
            [config.image_w / config.original_image_w],
            [config.image_h / config.original_image_h]
        ], dtype=depth.dtype)
    # Reshape to image
    warp = tf.transpose(warp, [0, 2, 1]) # [batch, h*w, 2]
    warp = tf.reshape(warp, [depth.shape[0], depth.shape[1], depth.shape[2], 2]) # [batch, h, w, 2]
    # Convert to tf warp format: swap x and y
    warp = warp[:,:,:,::-1]
    
    # Compute border mask
    border_mask_0 = (pix_coords_xy[:, 0] >= 0) & (pix_coords_xy[:, 0] < config.original_image_w)
    border_mask_1 = (pix_coords_xy[:, 1] >= 0) & (pix_coords_xy[:, 1] < config.original_image_h)
    border_mask = border_mask_0 & border_mask_1
    border_mask = tf.reshape(border_mask, [depth.shape[0], depth.shape[1], depth.shape[2]]) # [batch, h, w]

    return warp, cam_coords, border_mask

def compute_projection_mask_s2(warp):
    # Random image
    noise_channels = 10
    noise = tf.random.normal(shape=[warp.shape[0], config.image_h, config.image_w, noise_channels])
    noise = tfa.image.dense_image_warp(noise, warp)
    # Gradients
    noise_edges = tf.image.sobel_edges(noise)
    noise_edges_norm = tf.sqrt(tf.reduce_sum(noise_edges**2, 4, keepdims=True))
    noise_edges = noise_edges / noise_edges_norm
    # Average squared cosine between gradients on different channels
    mul_1 = tf.reshape(noise_edges, [-1, config.image_h, config.image_w, noise_channels, 1, 2])
    mul_2 = tf.reshape(noise_edges, [-1, config.image_h, config.image_w, 1, noise_channels, 2])
    mean_cos2 = tf.reduce_mean(
        tf.reduce_sum(mul_1 * mul_2, 5)**2,
        [3, 4])
    # Nematic order parameter S2
    s2 = 3/2 * mean_cos2 - 0.5
    # To binary mask
    mask = tf.cast(s2 > 0.9, tf.int32)
    # Apply dilation
    mask = tf.nn.dilation2d(
        mask[:,:,:,None],
        filters=tf.zeros((3,3,1), dtype=tf.int32),
        strides=(1,1,1,1),
        data_format='NHWC',
        dilations=(1,1,1,1),
        padding="SAME")[:,:,:,0]
    # Convert to float {0., 1.}
    mask = 1 - tf.cast(mask, tf.float32)
    return mask
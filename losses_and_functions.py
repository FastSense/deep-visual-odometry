'''
Fuctions used to calculate losses
'''

from load_config import *
from geometry_utils import *
from models import get_pretrained_vgg16

vgg16_features = get_pretrained_vgg16()

def random_augmentation(batch_size):
    aug = {}
    # Color jitter
    aug['color_shift'] = (tf.random.normal([batch_size, 1, 1, 1, 3]) * 2 - 1) * 0.1
    shift_bool = tf.cast(tf.random.normal([batch_size, 1, 1, 1, 1]) < 0.5, tf.float32)
    aug['color_shift'] = aug['color_shift'] * shift_bool
    # Flip
    if config.flip_augmentation:
        aug['flip_x'] = tf.cast(tf.random.normal([batch_size, 1, 1, 1, 1]) < 0.5, tf.float32)
        aug['flip_y'] = tf.cast(tf.random.normal([batch_size, 1, 1, 1, 1]) < 0.5, tf.float32)
    return aug

@tf.function
def augment(batch, aug):
    if batch is None:
        return None
    # Flip
    if 'flip_x' in aug:
        batch = batch * (1 - aug['flip_x']) + batch[:, :, :, ::-1] * aug['flip_x']
    if 'flip_y' in aug:
        batch = batch * (1 - aug['flip_y']) + batch[:, :, ::-1] * aug['flip_y']
    # Color jitter
    if 'color_shift' in aug:
        batch = batch + aug['color_shift']
        batch = tf.clip_by_value(batch, 0, 1)
    return batch

@tf.function
def blur_image(img, kernel_size=3):
    # Padding
    pad = kernel_size // 2
    paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
    img = tf.pad(img, paddings, 'REFLECT')
    
    # Pooling
    img = tf.nn.avg_pool2d(img, kernel_size, 1, 'VALID')
    
    return img

@tf.function
def compute_ssim(x, y, kernel_size=3, c1=0.01**2, c2=0.03**2):
    # Padding
    pad = kernel_size // 2
    paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
    x = tf.pad(x, paddings, 'REFLECT')
    y = tf.pad(y, paddings, 'REFLECT')
    
    # Pooling
    pool = lambda img: tf.nn.avg_pool2d(img, kernel_size, 1, 'VALID')
    mu_x = pool(x)
    mu_y = pool(y)
    
    sigma_x  = pool(x ** 2) - mu_x ** 2
    sigma_y  = pool(y ** 2) - mu_y ** 2
    sigma_xy = pool(x * y) - mu_x * mu_y
    
    SSIM_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    
    return tf.clip_by_value((1 - SSIM_n / SSIM_d) / 2, 0, 1)

@tf.function
def per_pixel_loss(img, restored, mask=1, blur=False):
    # Denoise images
    if blur:
        img = blur_image(img)
        restored = blur_image(restored)
    # Absolute difference
    l1_loss = tf.reduce_mean(tf.abs(img - restored), axis=3)
    # Semantic similarity
    ssim_loss = tf.reduce_mean(compute_ssim(img, restored), axis=3)
    # Combine
    pp = 0.85 * ssim_loss + 0.15 * l1_loss
    pp = tf.reduce_mean(pp * tf.cast(mask, pp.dtype))
    return pp

@tf.function
def get_smooth_loss(value, img):
    """
    Computes the smoothness loss for disparity or optical flow
    The color image is used for edge-aware smoothness
    """
    value = tf.cast(value, img.dtype)
    
    grad_x = tf.abs(value[:, :, :-1] - value[:, :, 1:])
    grad_y = tf.abs(value[:, :-1, :] - value[:, 1:, :])

    grad_img_x = tf.reduce_mean(tf.abs(img[:, :, :-1] - img[:, :, 1:]), 3, keepdims=True)
    grad_img_y = tf.reduce_mean(tf.abs(img[:, :-1, :] - img[:, 1:, :]), 3, keepdims=True)

    grad_x = grad_x * tf.exp(-grad_img_x)
    grad_y = grad_y * tf.exp(-grad_img_y)

    return tf.reduce_mean(grad_x) + tf.reduce_mean(grad_y)

@tf.function
def depth_consistency(depth, restored_depth, mask=1):
    con = tf.reduce_mean((tf.abs(1 / restored_depth - 1 / depth)) * mask)
    return tf.cast(con, tf.float32)

def calculate_loss_rigid(img_first, img_other, depth, pose, mask=None, training=True, epoch=-1, warmup=False):
    # Compute optical flow under rigid scene assumption
    warp, _, border_mask = get_warp(
        depth,
        pose,
        config_runtime.camera_matrix_l,
        config_runtime.inv_camera_matrix_l,
    )

    # Restore first image from the other image using pose and depth
    restored_rigid = tfa.image.dense_image_warp(img_other, -warp)
    border_mask = compute_projection_mask_s2(-warp) * tf.cast(border_mask, tf.float32)

    # Any additional mask (e.g. flow consistency mask)
    if mask is not None:
        border_mask = border_mask * mask

    # Image reconstruction loss
    pp = per_pixel_loss(
        img_first,
        restored_rigid,
        mask=border_mask,
        blur=warmup)
    
    # Depth smoothness loss
    smoothness_depth = get_smooth_loss(1 / depth[:,:,:,None], img_first)
    
    return warp, pp, smoothness_depth, border_mask

def calculate_loss_flow(img_first, img_other, flow, flow_mask, rigid_flow, training=True, epoch=-1, warmup=False):
    # Restore previous image from the left image using predicted optical flow
    restored_flow = tfa.image.dense_image_warp(img_other, -flow)

    # Compute consistency between flow and rigid scene movement
    rigid_flow = tf.cast(rigid_flow, flow.dtype)
    consistency = (tf.stop_gradient(flow) - rigid_flow)**2
    consistency = tf.reduce_mean(consistency, 3)
    moving_object_penalty = tfp.stats.percentile(
        consistency,
        config.moving_object_percentile,
        axis=[1,2],
        keepdims=True,
        preserve_gradients=False,
    )
    consistency = consistency * (1 - flow_mask) + moving_object_penalty * flow_mask
    flow_consistency = tf.reduce_mean(consistency)

    # Image reconstruction loss
    pp = per_pixel_loss(
        img_first,
        restored_flow,
        blur=warmup)

    # Flow smoothness loss
    smoothness_flow = get_smooth_loss(flow, img_first)

    return pp, smoothness_flow, flow_consistency

def calculate_loss(flow_net, depth_net, pose_net, rollout_l, rollout_r=None, training=True, epoch=-1, warmup=False):
    # Generate random augmentation
    aug = random_augmentation(rollout_l.shape[0])

    # Apply random image flips
    if training and config.flip_augmentation:
        aug_flip = {'flip_x': aug['flip_x'], 'flip_y': aug['flip_y']}
        rollout_l = augment(rollout_l, aug_flip)
        rollout_r = augment(rollout_r, aug_flip)
        # Swap right and left images
        rollout_l, rollout_r = (
            rollout_l * (1 - aug['flip_x']) + rollout_r * aug['flip_x'],
            rollout_r * (1 - aug['flip_x']) + rollout_l * aug['flip_x'],
        )

    # Save original images before color augmentation
    rollout_l_orig = rollout_l
    rollout_r_orig = rollout_r
    
    # Augmentation
    if training:
        aug_color = {'color_shift': aug['color_shift']}
        rollout_l = augment(rollout_l, aug_color)
        rollout_r = augment(rollout_r, aug_color)
    
    # Make predictions
    batch_size, rollout_size, w, h, c = rollout_l.shape
    new_shape = [batch_size * (rollout_size - 1), w, h, c]
    img_l_orig = tf.reshape(rollout_l_orig[:, 1:], new_shape)
    img_r_orig = tf.reshape(rollout_r_orig[:, 1:], new_shape)
    img_p_orig = tf.reshape(rollout_l_orig[:, :-1], new_shape)
    img_l = tf.reshape(rollout_l[:, 1:], new_shape)
    img_r = tf.reshape(rollout_r[:, 1:], new_shape)
    img_p = tf.reshape(rollout_l[:, :-1], new_shape)
    img_pr = tf.reshape(rollout_r[:, :-1], new_shape)

    # Predict optical flow at different scales
    flow_input = tf.concat([img_l, img_p], 3)
    flows = flow_net(flow_input, training=training)
    flow_mask = flows[-1][:,:,:,0]
    flows = flows[:-1]

    # Predict depth maps of different scales
    depth_input = tf.concat([img_l, img_r], 3)
    depths = depth_net(depth_input, training=training)
    depths = [tf.cast(depth, tf.float64) for depth in depths]
    
    # Predict pose
    pose_input = tf.concat([img_l, img_p, img_r, img_pr], 3)
    if config.rollout_size > 2 and config.lambda_skip_pose > 0:
        pose_input = tf.concat([
            pose_input,
            tf.concat([
                rollout_l[:, -1],
                rollout_l[:, 0],
                rollout_r[:, -1],
                rollout_r[:, 0],
            ], 3)
        ], 0)
    pose_predictions = pose_net(pose_input, training=training)
    pose_predictions = tf.cast(pose_predictions, tf.float64)
    if config.rollout_size > 2 and config.lambda_skip_pose > 0:
        pose_predictions_first_last = pose_predictions[-batch_size:]
        pose_predictions = pose_predictions[:-batch_size]
    prev2left_transform = rt_matrix_from_6_dof(pose_predictions) # [batch * (rollout - 1), 4, 4]
    # Poses for each scale
    poses = [prev2left_transform] + [tf.stop_gradient(prev2left_transform)] * (len(depths) - 1)

    # Initialize losses
    pp_sum, smoothness_flow_sum, smoothness_depth_sum, flow_consistency_sum = 0., 0., 0., 0.

    # Compute loss at different scales
    for flow, depth, pose in zip(flows, depths, poses):
        # Stereo reprojection loss
        if config.use_stereo:
            transform = tf.constant(config_runtime.right_left_transform, dtype=depth.dtype)
            _, pp_lr, _, _ = calculate_loss_rigid(
                img_l_orig, img_r_orig, depth, transform,
                training=training, epoch=epoch, warmup=warmup
            )
        
        # Rigid scene reprojection loss
        rigid_flow, pp_lp, smoothness_depth, border_mask = calculate_loss_rigid(
            img_l_orig, img_p_orig, depth, pose,
            mask=tf.stop_gradient(1 - flow_mask),
            training=training, epoch=epoch, warmup=warmup
        )

        # Optical flow loss
        pp_flow, smoothness_flow, flow_consistency = calculate_loss_flow(
            img_l_orig, img_p_orig, flow,
            flow_mask * tf.cast(border_mask, flow_mask.dtype),
            rigid_flow,
            training=training, epoch=epoch, warmup=warmup
        )

        # Image losses
        if config.use_stereo:
            pp_sum = pp_sum + pp_lr
        pp_sum = pp_sum + pp_lp + pp_flow
        # Depth smoothness loss
        smoothness_depth_sum = smoothness_depth_sum + smoothness_depth
        # Flow smoothness loss
        smoothness_flow_sum = smoothness_flow_sum + smoothness_flow
        # Flow consistency loss
        flow_consistency_sum = flow_consistency_sum + flow_consistency
    
    if config.rollout_size > 2 and config.lambda_skip_pose > 0:
        # Compute 6-dof transform from first to last frame in rollout
        pose = tf.reshape(prev2left_transform, [batch_size, rollout_size - 1, 4, 4])
        pose_0_to_last = pose[:, 0]
        for i in range(1, config.rollout_size - 1):
            pose_0_to_last = pose[:, i] @ pose_0_to_last
        # Get the same transform predicted by pose net
        pose_0_to_last_pred = rt_matrix_from_6_dof(pose_predictions_first_last) # [-1, 4, 4]
        # Get positions
        xyz_1 = pose_0_to_last[:, :-1, 3] # [-1, 3]
        xyz_2 = pose_0_to_last_pred[:, :-1, 3]
        # Get small angles
        angles_1 = tf.concat([
            pose_0_to_last[:, 0, 1:3],
            pose_0_to_last[:, 1, 3:]
        ], 1) # [-1, 3]
        angles_2 = tf.concat([
            pose_0_to_last_pred[:, 0, 1:3],
            pose_0_to_last_pred[:, 1, 3:]
        ], 1) # [-1, 3]
        # Compute difference between them
        loss_skip = (xyz_1 - xyz_2)**2 + (angles_1 - angles_2)**2
        # loss_skip = tf.math.abs(xyz_1 - xyz_2) + tf.math.abs(angles_1 - angles_2)
        loss_skip = tf.reduce_mean(loss_skip)
        loss_skip = tf.cast(loss_skip, tf.float32)

    # Summary Loss
    smoothness = smoothness_depth_sum + smoothness_flow_sum
    l_sm_flow = config.lambda_smoothness_flow
    l_sm_depth = config.lambda_smoothness_depth
    lambda_f = config.lambda_flow_consistency
    loss = (pp_sum + l_sm_flow * smoothness_flow_sum + l_sm_depth * smoothness_depth_sum + lambda_f * flow_consistency_sum) / len(depths)
    if config.rollout_size > 2 and config.lambda_skip_pose > 0:
        loss = loss + loss_skip * config.lambda_skip_pose

    if tf.math.is_nan(loss):
        tf.print('Error! Nan loss')
    
    return {
        'loss': loss,
        'per_pixel': pp_sum,
        'smoothness': smoothness,
        'flow_consistency': flow_consistency_sum,
    }
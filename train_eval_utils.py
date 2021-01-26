'''
Functions which are used in the training loop and in the evaluation.
'''

from load_config import *
from losses_and_functions import *
from data_utils import *
from kitti_odometry import KittiEvalOdom

to_log = ['loss', 'per_pixel', 'smoothness', 'flow_consistency']

@tf.function
def predict_step(pose_net, img_l, img_p, img_r, img_pr):
    pose_input = tf.concat([img_l, img_p, img_r, img_pr], 3)
    transform = pose_net(pose_input, training=False)
    return transform

def predict_poses(pose_net, seq_num, fine_tune=False, depth_net=None):
    # Initialize metrics
    poses = []
    poses_tuned = []

    # Initialize data generator
    data_generator = BatchGenerator([seq_num])

    # Evaluation
    for batch in tqdm(data_generator):
        rollout_l, rollout_r = batch
        img_l = rollout_l[:, 1]
        img_p = rollout_l[:, 0]
        img_r = rollout_r[:, 1]
        img_pr = rollout_r[:, 0]
        transform = predict_step(pose_net, img_l, img_p, img_r, img_pr)
        # Add poses
        poses.append(transform.numpy())
        # Fine tune this transform vector
        if fine_tune:
            transform = tf.Variable(transform.numpy(), trainable=True)
            depth = depth_net(tf.concat([img_l, img_r], 3), training=False)[0]
            optimizer = tf.keras.optimizers.Adam(lr=0.001)
            for i in range(20):
                with tf.GradientTape() as tape:
                    warp, cam_coords, _ = get_warp(
                        depth,
                        rt_matrix_from_6_dof(transform),
                        tf.cast(config_runtime.camera_matrix_l, depth.dtype),
                        tf.cast(config_runtime.inv_camera_matrix_l, depth.dtype),
                    )
                    restored_p = tfa.image.dense_image_warp(img_p, -warp)
                    loss = per_pixel_loss(img_l, restored_p)
                gradient_t = tape.gradient(loss, transform)
                optimizer.apply_gradients([(gradient_t, transform)])
            poses_tuned.append(transform.numpy())
    
    poses = np.concatenate(poses, 0)
    
    if not fine_tune:
        return poses
    else:
        poses_tuned = np.concatenate(poses_tuned, 0)
        return poses, poses_tuned

@tf.function
def train_step(batch, flow_net, depth_net, pose_net, optimizer, trainable_variables, metrics, epoch, warmup):
    # Forward pass
    with tf.GradientTape() as tape:
        results = calculate_loss(flow_net, depth_net, pose_net, *batch, training=True, epoch=epoch, warmup=warmup)
        loss = results['loss']
    
    # Average everything
    for name in to_log:
        metrics[name](results[name])

    # Backward pass
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return results

def train_epoch(flow_net, depth_net, pose_net, optimizer, data_generator, epoch, use_wandb=True):
    # Initialize metrics
    metrics = {
        name: tf.keras.metrics.Mean(name=name)
        for name in to_log
    }
    
    # Training steps
    trainable_variables = depth_net.trainable_variables + pose_net.trainable_variables + flow_net.trainable_variables
    # bg_data_generator = BackgroundGenerator(data_generator)
    for batch in tqdm(data_generator, total=len(data_generator)):
        results = train_step(batch, flow_net, depth_net, pose_net, optimizer, trainable_variables, metrics, epoch, config_runtime.blur_before_loss)
        if use_wandb:
            wandb.log({key: results[key].numpy() for key in results})
        if config.debug_mode:
            break

    # Combine results
    result = {
        name: metrics[name].result().numpy()
        for name in to_log
    }

    return result

def eval_odometry(seq, seq_pred):
    eval_odom = KittiEvalOdom()

    poses_gt = {i: seq.poses[i] for i in range(len(seq.poses))}
    poses_result = {i: seq_pred.poses[i] for i in range(len(seq_pred.poses))}
    
    result = {}
    
    # compute sequence errors
    seq_err = eval_odom.calc_sequence_errors(poses_gt, poses_result)

    # Compute segment errors
    avg_segment_errs = eval_odom.compute_segment_error(seq_err)

    # compute overall error
    ave_t_err, ave_r_err = eval_odom.compute_overall_err(seq_err)
    result['t_err(%)'] = ave_t_err*100
    result['r_err(deg_per_100m)'] = ave_r_err/np.pi*180*100

    # Compute ATE
    ate = eval_odom.compute_ATE(poses_gt, poses_result)
    result['ATE(m)'] = ate

    # Compute RPE
    rpe_trans, rpe_rot = eval_odom.compute_RPE(poses_gt, poses_result)
    result['RPE(m)'] = rpe_trans
    result['RPE(deg)'] = rpe_rot * 180 /np.pi
    
    return result
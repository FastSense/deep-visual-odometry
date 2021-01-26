'''
This is the main file.
Example command:

$ WANDB_NAME='test run' CUDA_VISIBLE_DEVICES=0 python3 train.py config_kitti.yaml

This will run the full training of the model on the specified dataset.
Results will be saved in the wandb directory and in cloud.
'''

import load_config
load_config.initialize()

from load_config import *
from data_utils import *
from models import *
from train_eval_utils import *
from plot_utils import *

# Data generator
train_generator = BatchGenerator(config.train_sequences, shuffle=True)
# Optimizer
learning_rate = tf.Variable(config.learning_rate, dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

# Models
flow_net = FlowNet()
depth_net = DepthNet()
pose_net = PoseNet()

# Load weights for debugging
if config.load_weights:
    print('Loading weights from:', config.load_weights_from)
    flow_net.load_weights(os.path.join(config.load_weights_from, 'flow_net.h5'))
    depth_net.load_weights(os.path.join(config.load_weights_from, 'depth_net.h5'))
    pose_net.load_weights(os.path.join(config.load_weights_from, 'pose_net.h5'))

flow_net.compile()
depth_net.compile()
pose_net.compile()

flow_net.summary()
depth_net.summary()
pose_net.summary()

# Show initial depth and flow
for seq in sorted(list(set(config.dev_sequences + config.train_sequences))):
    fig = plot_depth_flow(depth_net, flow_net, seq)
    wandb.log({
        'depth_initial_{}'.format(seq): plt
    })
    plt.close()

config_runtime.blur_before_loss = config.blur_before_loss

# Jointly train flow, depth, pose
for epoch in range(1, config.num_epochs + 1):
    print('Epoch', epoch)
    train_log = train_epoch(flow_net, depth_net, pose_net, optimizer, train_generator, epoch)
    print('Training log:\n\t{}'.format(train_log))

    if epoch == config.lr_switch_after:
        new_lr = config.learning_rate * 0.1
        print('Changing learning rate to', new_lr)
        learning_rate.assign(new_lr)
    
    if epoch == config.remove_blur_after:
        config_runtime.blur_before_loss = False

    # Show depth and flow
    for seq in config.dev_sequences:
        fig = plot_depth_flow(depth_net, flow_net, seq)
        wandb.log({
            'depth_epoch{}_{}'.format(epoch, seq): plt
        })
        plt.close()

    if config.debug_mode:
        break

    # Evaluate on dev sequences
    avg_metrics = {}
    for i, seq_id in enumerate(config.dev_sequences):
        # Predict poses
        poses = predict_poses(pose_net, seq_id)
        seq = OdoSequence(seq_id)
        seq_pred = OdoSequence(seq_id)
        predictions = [np.eye(4)]
        for transform in rt_matrix_from_6_dof(poses):
            transform = transform.numpy().astype('float32')
            predictions.append(predictions[-1] @ transform)
        seq_pred.poses = predictions
        seq_pred.save(os.path.join(wandb.run.dir, f'predictions_{seq_id}_epoch{epoch}.txt'))
        # Draw sequences
        log = {}
        for view in config.plot_views:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.annotate(
                seq_id,
                xy=(0, 1), xytext=(12, -12), va='top',
                xycoords='axes fraction', textcoords='offset points',
                bbox=dict(facecolor='none', edgecolor='black'))
            seq.plot_poses(ax=ax, view=view, label='Ground truth', color='black', ls='--')
            seq_pred.plot_poses(ax=ax, view=view, label='Prediction', alpha=0.8)
            ax.legend(loc="lower right")
            ax.set_xlabel(f'${view[0]}$ (m)')
            ax.set_ylabel(f'${view[1]}$ (m)')
            name = '{}_{}_epoch{}'.format(seq_id, view, epoch)
            log[name] = wandb.Image(plt)
        # Evaluate odometry
        eval_odom = eval_odometry(seq, seq_pred)
        for key in eval_odom:
            avg_metrics[key] = avg_metrics.get(key, []) + [eval_odom[key]]
        # Log
        log.update({
            '{} {}'.format(seq_id, key): eval_odom[key]
            for key in eval_odom
        })
        pprint(log)
        wandb.log(log)
        plt.close()
    # Average metrics
    wandb.log({
        'Dev ' + key: np.mean(avg_metrics[key]) for key in avg_metrics
    })

# Save input and output examples
seq = OdoSequence(config.dev_sequences[0])
img_l = tf.constant(imread(seq.images_l[101])[None], dtype=tf.float32)
img_r = tf.constant(imread(seq.images_r[101])[None], dtype=tf.float32)
img_p = tf.constant(imread(seq.images_l[100])[None], dtype=tf.float32)
img_pr = tf.constant(imread(seq.images_r[100])[None], dtype=tf.float32)
# Predict depth
depth_inp = tf.concat([img_l, img_r], 3)
depth_out = depth_net(depth_inp, training=False)
# Predict pose
pose_inp = tf.concat([img_l, img_p, img_r, img_pr], 3)
pose_out = pose_net(pose_inp, training=False)
# Save to numpy files
np.save(os.path.join(wandb.run.dir, 'ut_depth_inp.npy'), depth_inp.numpy())
np.save(os.path.join(wandb.run.dir, 'ut_depth_out.npy'), depth_out[0].numpy())
np.save(os.path.join(wandb.run.dir, 'ut_pose_inp.npy'), pose_inp.numpy())
np.save(os.path.join(wandb.run.dir, 'ut_pose_out.npy'), pose_out.numpy())

# Print summary the second time
pose_net.summary()

# Save models
print('Saving flow net')
flow_net.save(os.path.join(wandb.run.dir, 'flow_net'))
flow_net.save(os.path.join(wandb.run.dir, 'flow_net.h5'))
print('Saving depth net')
depth_net.save(os.path.join(wandb.run.dir, 'depth_net'))
depth_net.save(os.path.join(wandb.run.dir, 'depth_net.h5'))
print('Saving pose net')
pose_net.save(os.path.join(wandb.run.dir, 'pose_net'))
pose_net.save(os.path.join(wandb.run.dir, 'pose_net.h5'))

# Save tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(pose_net)
tflite_model = converter.convert()
open(os.path.join(wandb.run.dir, 'pose_net.fp32.tflite'), "wb").write(tflite_model)

# Save quantized tflite model
for mode in ['int8', 'int8_iofp32', 'int8_iouint8']:
    # converter = tf.lite.TFLiteConverter.from_keras_model(pose_net)
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(os.path.join(wandb.run.dir, 'pose_net.h5'))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.target_spec.supported_types = [tf.int8]
    def representative_dataset_gen():
        for seq_id in config.train_sequences:
            seq = OdoSequence(seq_id)
            for i in tqdm(range(1, len(seq), 100)):
                # Get sample input data as a numpy array in a method of your choosing.
                img_l = imread(seq.images_l[i])
                img_p = imread(seq.images_l[i - 1])
                img_r = imread(seq.images_r[i])
                img_pr = imread(seq.images_r[i - 1])
                inp = np.concatenate([img_l, img_p, img_r, img_pr], 2)[None].astype('float32')
                yield [inp]
    converter.representative_dataset = representative_dataset_gen
    if mode == 'int8':
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif mode == 'int8_iouint8':
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    print('Quantizing posenet')
    tflite_quant_model = converter.convert()
    open(os.path.join(wandb.run.dir, f'pose_net.{mode}.tflite'), "wb").write(tflite_quant_model)

    # Convert to edge TPU
    command = 'edgetpu_compiler -s {} -o {}'.format(
        os.path.join(wandb.run.dir, f'pose_net.{mode}.tflite'),
        wandb.run.dir,
    )
    print('Executing command:')
    print(command)
    os.system(command)

# Convert saved models to onnx
convert_script = 'CUDA_VISIBLE_DEVICES="" python3 -m tf2onnx.convert --saved-model {} --output {} --opset {}'
print('Converting pose_net to onnx')
os.system(convert_script.format(
    os.path.join(wandb.run.dir, 'pose_net'),
    os.path.join(wandb.run.dir, 'pose_net.op10.onnx'),
    10,
))
os.system(convert_script.format(
    os.path.join(wandb.run.dir, 'pose_net'),
    os.path.join(wandb.run.dir, 'pose_net.op11.onnx'),
    11,
))
# print('Converting depth_net to onnx')
# os.system(convert_script.format(
#     os.path.join(wandb.run.dir, 'depth_net'),
#     os.path.join(wandb.run.dir, 'depth_net.op10.onnx'),
#     10,
# ))
# os.system(convert_script.format(
#     os.path.join(wandb.run.dir, 'depth_net'),
#     os.path.join(wandb.run.dir, 'depth_net.op11.onnx'),
#     11,
# ))

if config.debug_mode:
    exit()

# Evaluate on all sequences
for i, seq_id in enumerate(config.dev_sequences + config.test_sequences):
    # Predict poses
    poses, poses_tuned = predict_poses(pose_net, seq_id, fine_tune=True, depth_net=depth_net)
    seq = OdoSequence(seq_id)
    seq_pred = OdoSequence(seq_id)
    seq_pred_tuned = OdoSequence(seq_id)
    # Integrate predictions
    for _seq, _poses in [(seq_pred, poses), (seq_pred_tuned, poses_tuned)]:
        predictions = [np.eye(4)]
        for transform in rt_matrix_from_6_dof(_poses):
            transform = transform.numpy().astype('float32')
            predictions.append(predictions[-1] @ transform)
        _seq.poses = predictions
    # Save predicted poses to file
    seq_pred.save(os.path.join(wandb.run.dir, f'predictions_{seq_id}_final.txt'))
    seq_pred_tuned.save(os.path.join(wandb.run.dir, f'predictions_tuned_{seq_id}_final.txt'))
    # Draw sequences
    log = {}
    for view in config.plot_views:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.annotate(
            'Seq. ' + seq_id,
            xy=(0, 1), xytext=(12, -12), va='top',
            xycoords='axes fraction', textcoords='offset points',
            bbox=dict(facecolor='none', edgecolor='black'))
        seq.plot_poses(ax=ax, view=view, label='Ground truth', color='black', ls='--')
        seq_pred.plot_poses(ax=ax, view=view, label='Prediction', alpha=0.8)
        seq_pred_tuned.plot_poses(ax=ax, view=view, label='Prediction+GD', alpha=0.8)
        ax.legend(loc="lower right")
        ax.set_xlabel(f'${view[0]}$ (m)')
        ax.set_ylabel(f'${view[1]}$ (m)')
        name = '{}_{}_final'.format(seq_id, view, epoch)
        log[name] = wandb.Image(plt)
    # Evaluate odometry
    eval_odom = eval_odometry(seq, seq_pred)
    eval_odom_tuned = eval_odometry(seq, seq_pred_tuned)
    log.update({
        '{} {}'.format(seq_id, key): eval_odom[key]
        for key in eval_odom
    })
    log.update({
        '{} (+GD) {}'.format(seq_id, key): eval_odom_tuned[key]
        for key in eval_odom_tuned
    })
    # Log
    wandb.log(log)
    plt.close()

print('Program finished')
import load_config
config, config_runtime = load_config.initialize(use_wandb=False)

from data_utils import OdoSequence, imread
from train_eval_utils import predict_step
from geometry_utils import rt_matrix_from_6_dof, get_warp
from plot_utils import flow_to_rgb

import numpy as np
import tensorflow as tf
import io
from tqdm import tqdm
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors

import warnings
warnings.filterwarnings("ignore")

flow_net = tf.keras.models.load_model(os.path.join(config.load_weights_from, 'flow_net'))
depth_net = tf.keras.models.load_model(os.path.join(config.load_weights_from, 'depth_net'))
pose_net = tf.keras.models.load_model(os.path.join(config.load_weights_from, 'pose_net'))

view = 'xz'

# Evaluate on all sequences
for seq_id in sorted(list(set(config.dev_sequences + config.test_sequences))):
    # Open video writer
    fps = config.frames_per_second * (config.skip_frames_predict + 1)
    videodims = (1280, 720)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(f'film_{seq_id}.avi', fourcc, fps, videodims)

    seq = OdoSequence(seq_id)
    seq_pred = OdoSequence(seq_id)
    poses = []
    
    # Step between frames
    delta = config.skip_frames_predict + 1
    
    first_frame = delta
    last_frame = len(seq)
    # first_frame = len(seq)*60//110
    # last_frame = len(seq)*70//110

    for i in tqdm(range(first_frame, last_frame, delta)):
        img_l = tf.constant(imread(seq.images_l[i])[None], dtype=tf.float32)
        img_r = tf.constant(imread(seq.images_r[i])[None], dtype=tf.float32)
        img_p = tf.constant(imread(seq.images_l[i - delta])[None], dtype=tf.float32)

        flows = predict_step(flow_net, img_l, img_p)
        depths = predict_step(depth_net, img_l, img_r)
        transform_prediction = predict_step(pose_net, img_l, img_p)
        # Add poses
        poses.append(transform_prediction.numpy())

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        # Draw images
        axes[0, 0].set_title('Left image')
        axes[0, 0].imshow(img_l[0].numpy().astype(float))
        axes[0, 1].set_title('Right image')
        axes[0, 1].imshow(img_r[0].numpy().astype(float))
        # Draw sequence
        predictions = [np.eye(4)]
        for transform in rt_matrix_from_6_dof(np.concatenate(poses, 0)):
            transform = transform.numpy().astype('float32')
            predictions.append(predictions[-1] @ transform)
        seq_pred.poses = predictions
        ax = axes[0, 2]
        ax.annotate(
            'Seq. ' + seq_id,
            xy=(0, 1), xytext=(12, -12), va='top',
            xycoords='axes fraction', textcoords='offset points',
            bbox=dict(facecolor='none', edgecolor='black'))
        seq.plot_poses(ax=ax, view=view, label='Ground truth', color='black', ls='--')
        seq_pred.plot_poses(ax=ax, view=view, label='Prediction', alpha=0.8)
        ax.legend(loc="lower right")
        ax.set_xlabel(f'${view[0]}$ (m)')
        ax.set_ylabel(f'${view[1]}$ (m)')
        plt.tight_layout()
        
        # Draw depth
        ax = axes[1, 0]
        ax.set_title('Disparity map ($\\frac{1}{depth}$)')
        im = ax.imshow(
            1 / depths[0][0].numpy().astype(float),
            vmin=0,
            vmax=1 / config.min_depth_on_plot,
        )
        cax = fig.add_axes([0.33, 0.1, 0.005, 0.3])
        plt.colorbar(im, cax=cax, orientation='vertical')
        
        # Draw flow
        ax = axes[1, 1]
        ax.set_title('Optical flow')
        im = ax.imshow(
            flow_to_rgb(flows[0][0].numpy().astype(float))
        )

        # Moving objects mask
        ax = axes[1, 2]
        ax.set_title('Moving objects mask')
        im = ax.imshow(
            flows[-1][0,:,:,0].numpy().astype(float),
            vmin=0,
            vmax=1,
            cmap='gray',
        )

        # Plt to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, videodims)
        video.write(frame)
        plt.close()

    video.release()
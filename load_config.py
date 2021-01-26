'''
Very important file.
This file is imported in every other file.
initialize() must be called from train.py

Functions here are responsible for loading config from .yaml file
'''

import yaml
import os
import sys
from argparse import Namespace
from tqdm import tqdm
from multiprocessing import Process, Queue, Pool
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
import gc
from pprint import pprint
from prefetch_generator import BackgroundGenerator, background

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers as kl
from tensorflow.python.ops import image_ops_impl

import logging

# Init wandb
import wandb

def initialize(config_init=None, use_wandb=True):
    global config, config_runtime

    if config_init is not None:
        config = config_init
    else:
        # Load config from file
        with open(sys.argv[1]) as f:
            config = Namespace()
            config.__dict__ = yaml.safe_load(f)

    if use_wandb:
        wandb.init(project="vo_research", sync_tensorboard=True, config=config)

    # Configure tensorflow
    # Disable warnings
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # Do not allow tf to take the entire GPU memory
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    # XLA for faster training
    tf.config.optimizer.set_jit(True)

    # Fix the random seeds
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)


    # Transformation 4x4 matrix from right to left camera
    left_right_transform = np.array(
    [
        [1, 0, 0, config.distance_btw_cameras],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    ).astype(np.float32)
    right_left_transform = np.linalg.inv(left_right_transform)

    # Left camera intrinsic matrix
    camera_matrix_l = np.array(config.camera_matrix_l).astype(np.float64)
    camera_matrix_l = np.concatenate([camera_matrix_l, np.zeros_like(camera_matrix_l)[:,:1]], 1)
    camera_matrix_l = np.concatenate([camera_matrix_l, np.zeros_like(camera_matrix_l)[:1,:]], 0)
    camera_matrix_l[3, 3] = 1
    inv_camera_matrix_l = np.linalg.inv(camera_matrix_l)

    # Right camera intrinsic matrix
    camera_matrix_r = np.array(config.camera_matrix_r).astype(np.float64)
    camera_matrix_r = np.concatenate([camera_matrix_r, np.zeros_like(camera_matrix_r)[:,:1]], 1)
    camera_matrix_r = np.concatenate([camera_matrix_r, np.zeros_like(camera_matrix_r)[:1,:]], 0)
    camera_matrix_r[3, 3] = 1
    inv_camera_matrix_r = np.linalg.inv(camera_matrix_r)

    # Virtual camera matrix
    if config.use_virtual_camera:
        # Make virtual K
        virtual_camera_matrix = np.array(config.virtual_camera_matrix).astype(np.float64)
        virtual_camera_matrix = np.concatenate([virtual_camera_matrix, np.zeros_like(virtual_camera_matrix)[:,:1]], 1)
        virtual_camera_matrix = np.concatenate([virtual_camera_matrix, np.zeros_like(virtual_camera_matrix)[:1,:]], 0)
        virtual_camera_matrix[3, 3] = 1
        inv_virtual_camera_matrix = np.linalg.inv(virtual_camera_matrix)
        # Compute affine transform
        affine_transform_r = (virtual_camera_matrix[:3, :3] @ inv_camera_matrix_r[:3, :3])[:2]
        affine_transform_l = (virtual_camera_matrix[:3, :3] @ inv_camera_matrix_l[:3, :3])[:2]
        # Modify camera matrices
        camera_matrix_l = virtual_camera_matrix
        camera_matrix_r = virtual_camera_matrix
        inv_camera_matrix_l = inv_virtual_camera_matrix
        inv_camera_matrix_r = inv_virtual_camera_matrix

    # Save to the global variable
    config_runtime = Namespace()
    config_runtime.left_right_transform = left_right_transform
    config_runtime.right_left_transform = right_left_transform
    config_runtime.camera_matrix_l = camera_matrix_l
    config_runtime.camera_matrix_r = camera_matrix_r
    config_runtime.inv_camera_matrix_l = inv_camera_matrix_l
    config_runtime.inv_camera_matrix_r = inv_camera_matrix_r
    if config.use_virtual_camera:
        config_runtime.affine_transform_r = affine_transform_r
        config_runtime.affine_transform_l = affine_transform_l


    return config, config_runtime
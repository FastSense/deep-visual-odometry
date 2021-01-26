'''
This file is not used in training.
It is for testing the saved onnx file.
'''

import sys
import os
import numpy as np
import time

import onnxruntime as ort

path = sys.argv[1]


print('Pose:')
pose_net = ort.InferenceSession(os.path.join(path, 'pose_net.op11.onnx'))

input_name = pose_net.get_inputs()[0].name

pose_inp = np.load(os.path.join(path, 'ut_pose_inp.npy'))
pose_out = np.load(os.path.join(path, 'ut_pose_out.npy'))

start = time.time()
pose_out_rt = pose_net.run(None, {input_name: pose_inp})[0]
print('time:', time.time() - start)
print(pose_out_rt, '- out')
print(pose_out, '- reference')
print('Pose net allclose:', np.allclose(pose_out, pose_out_rt, rtol=0.001))
print()

print('Depth:')
depth_net = ort.InferenceSession(os.path.join(path, 'depth_net.op10.onnx'))

input_name = depth_net.get_inputs()[0].name

depth_inp = np.load(os.path.join(path, 'ut_depth_inp.npy'))
depth_out = np.load(os.path.join(path, 'ut_depth_out.npy'))

start = time.time()
depth_out_rt = depth_net.run(None, {input_name: depth_inp})[0]
print('time:', time.time() - start)
print(depth_out_rt.shape, '- out shape')
print(depth_out.shape, '- reference shape')
print('Depth net allclose:', np.allclose(depth_out, depth_out_rt, rtol=0.001))
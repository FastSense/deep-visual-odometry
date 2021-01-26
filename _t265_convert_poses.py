import load_config
load_config.initialize(use_wandb=False)

from load_config import *
from data_utils import *
from geometry_utils import *

for seq_id in set(config.train_sequences + config.dev_sequences + config.test_sequences):
    print(seq_id)
    seq = OdoSequence(seq_id)
    
    path = os.path.join(config.data_path, 'sequences', seq_id)
    try:
        times = [float(l.strip()) for l in open(os.path.join(path, 'times_left.txt'))]
    except:
        times = [float(l.strip()) for l in open(os.path.join(path, 'times.txt'))]
    times.append(times[-1]*2 - times[-2])
    times = np.array(times)
    
    poses = [l.strip().replace(',',' ').split() for l in open(os.path.join(path, 'pose.txt'))]
    poses = [p for p in poses if len(p) == 7]
    poses = np.array(poses).astype(float)
    frame_poses = []
    for t in times:
        idx = np.argmin(np.abs(poses[:, 0] - t))
        frame_poses.append(poses[idx, [4, 5, 6, 1, 2, 3]])
    frame_poses = np.array(frame_poses)
    frame_poses = rt_matrix_from_6_dof(frame_poses).numpy()

    m_inv = np.linalg.inv(frame_poses[0])
    frame_poses = np.array([m_inv @ pose for pose in frame_poses])
    
    seq.poses = frame_poses
    
    # plt.figure(figsize=(14, 8))
    # for i, view in enumerate(config.plot_views):
    #     ax = plt.subplot(1, len(config.plot_views), i + 1)
    #     seq.plot_poses(view=view, ax=ax)
    # plt.show()
    
    pose_path = os.path.join(config.data_path, 'poses', seq_id + '.txt')
    seq.save(pose_path)
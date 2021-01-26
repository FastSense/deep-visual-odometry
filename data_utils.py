'''
Functions and classes that are used for loading data and generating batches.
'''

from load_config import *

def imread(path):
    img = cv2.imread(path)[:,:,::-1]

    # Affine transform
    if config.use_virtual_camera:
        rows, cols, _ = img.shape
        if 'image_2' in path:
            M = config_runtime.affine_transform_l
        elif 'image_3' in path:
            M = config_runtime.affine_transform_r
        img = cv2.warpAffine(img, M, (cols, rows))

    # Resize
    img = cv2.resize(img, (config.image_w, config.image_h)) / 255

    # Grayscale
    if config.grayscale_load:
        img = np.tile(img.mean(2, keepdims=True), [1, 1, 3])

    # Average and scale
    # for channel in range(3):
    #     img_c = img[:, :, channel]
    #     mean = img_c.mean()
    #     std = img_c.std()
    #     img[:, :, channel] = (img_c - mean) / (std * 3 + 1e-6) + 0.3

    return img

class OdoSequence:
    def __init__(self, seq_number):
        # Image paths
        img_path = os.path.join(config.data_path, 'sequences', seq_number)
        left_path = os.path.join(img_path, 'image_2')
        right_path = os.path.join(img_path, 'image_3')
        # Left and rigth images arrays
        self.images_l = [os.path.join(left_path, p) for p in sorted(os.listdir(left_path))]
        if config.use_stereo:
            self.images_r = [os.path.join(right_path, p) for p in sorted(os.listdir(right_path))]
        else:
            self.images_r = self.images_l
        
        # Read and parse the poses
        # https://github.com/alexkreimer/odometry/blob/master/devkit/readme.txt
        pose_file = os.path.join(config.data_path, 'poses', seq_number + '.txt')
        if os.path.isfile(pose_file):
            self.poses = []
            for line in open(pose_file):
                T_w_cam0 = np.fromstring(line.strip(), dtype=float, sep=' ')
                T_w_cam0 = T_w_cam0.reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                self.poses.append(T_w_cam0)
            self.poses = np.array(self.poses)
    
    def __len__(self):
        return len(self.images_l)
    
    def plot_poses(self, ax=None, view='xz', **kwargs):
        # Transform from camera coords to world coords
        coords = self.poses @ np.array([0,0,0,1])
        mx, my, mz = coords.T[:3]
        
        # Plot
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_aspect('equal')
        view_dict = dict(x=mx, y=my, z=mz)
        ax.plot(view_dict[view[0]], view_dict[view[1]], **kwargs)
        
        return ax
    
    def save(self, path):
        with open(path, 'w') as f:
            for pose in self.poses:
                s = ' '.join(pose[:3, :4].astype(str).flatten())
                f.write(s + '\n')


class BatchGenerator:
    def __init__(self, seq_numbers, shuffle=False):
        self.shuffle = shuffle
        # Load sequences
        self.sequences = [OdoSequence(i) for i in seq_numbers]
        self.image_paths = []
        sum_len = sum(map(len, self.sequences))
        self.ids = list(range(1,
                sum_len + config.batch_size - 1,
                config.batch_size
            ))
        self.ids[-1] = sum_len
    
    def get_batch(self, batch_idx):
        np.random.seed()
        rollouts_l, rollouts_r = [], []
        for i in range(self.ids[batch_idx], self.ids[batch_idx + 1]):
            if self.shuffle:
                # Sample a random pair of stereo pairs
                seq = np.random.choice(self.sequences)
                di = -1 - np.random.randint(config.max_frames_skip + 1)
                if config.reverse_runs and np.random.random() < 0.5:
                    di = -di
                di_full = di * config.rollout_size
                idx = np.random.randint(len(seq) - abs(di_full)) + max(0, -di_full)
                # Construct rollouts
                rollout_l = [
                    seq.images_l[j]
                    for j in range(idx, idx + di_full, di)
                ]
                rollout_r = [
                    seq.images_r[j]
                    for j in range(idx, idx + di_full, di)
                ]
            else:
                seq = self.sequences[0]
                # Get the subsequent images
                rollout_l = [
                    seq.images_l[i - 1],
                    seq.images_l[i],
                ]
                rollout_r = [
                    seq.images_r[i - 1],
                    seq.images_r[i],
                ]
            # Read images
            rollout_l = list(map(imread, rollout_l))
            rollouts_l.append(rollout_l)
            if config.use_stereo:
                rollout_r = list(map(imread, rollout_r))
                rollouts_r.append(rollout_r)
        # Convert batch to numpy
        rollouts_l = np.array(rollouts_l) # [batch, rollout, h, w, c]
        if config.use_stereo:
            rollouts_r = np.array(rollouts_r) # [batch, rollout, h, w, c]
        else:
            rollouts_r = rollouts_l
        # Return the batch
        return [rollouts_l, rollouts_r]

    @staticmethod
    def _batch_to_tf(batch):
        return [tf.constant(img, dtype=tf.float32) for img in batch]

    def __iter__(self):
        # Load batches in the background
        @background(max_prefetch=4)
        def generator():
            for i in range(len(self)):
                batch = self.get_batch(i)
                yield batch
        # Convert to tf in the main thread
        for batch in generator():
            yield self._batch_to_tf(batch)
    
    def __len__(self):
        if self.shuffle:
            return (len(self.ids) - 1) // (config.rollout_size - 1)
        else:
            return len(self.ids) - 1
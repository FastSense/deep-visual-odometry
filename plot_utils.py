from load_config import *
from data_utils import *

import warnings
warnings.filterwarnings("ignore")

def plot_depth_flow(depth_net, flow_net, seq_name):
    # Get example images
    seq = OdoSequence(seq_name)
    idx = len(seq) // 2
    img_l = tf.constant(imread(seq.images_l[idx])[None], dtype=tf.float32)
    img_r = tf.constant(imread(seq.images_r[idx])[None], dtype=tf.float32)
    img_p = tf.constant(imread(seq.images_l[idx - 1])[None], dtype=tf.float32)

    # Predict depth and flow at different scales
    flows = flow_net(tf.concat([img_l, img_p], 3), training=False)
    depths = depth_net(tf.concat([img_l, img_r], 3), training=False)

    # Draw figures
    n_rows = 1 + len(depths)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 8 * n_rows * config.image_h / config.image_w))
    # Draw input images
    axes[0, 0].imshow(img_l[0].numpy().astype(float))
    axes[0, 0].set_title('Left image')
    # axes[0, 1].imshow(img_r[0].numpy().astype(float))
    # axes[0, 1].set_title('Right image')
    axes[0, 1].imshow(flows[-1][0,:,:,0].numpy().astype(float), vmin=0, vmax=1, cmap='gray')
    axes[0, 1].set_title('Moving objects mask')

    # Draw depths and flows
    for i in range(len(depths)):
        depth = depths[i]
        flow = flows[i]
        plt.title('Scale $\\frac{{1}}{{{}}}$'.format(2**i))
        # Draw depth
        axes[i + 1, 0].set_title('Disparity map at scale $\\frac{{1}}{{{}}}$'.format(2**i))
        im = axes[i + 1, 0].imshow(
            1 / depth[0].numpy().astype(float),
            vmin=0,
            vmax=1 / config.min_depth_on_plot,
        )
        # Draw flow
        axes[i + 1, 1].set_title('Flow at scale $\\frac{{1}}{{{}}}$'.format(2**(i + 1)))
        im = axes[i + 1, 1].imshow(
            flow_to_rgb(flow[0].numpy().astype(float))
        )
    
    plt.tight_layout()
    
    return fig


def flow_to_rgb(flow):
    '''
    Function from this answer
    https://stackoverflow.com/a/49636438
    '''

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros([flow.shape[0], flow.shape[1], 3], dtype=np.uint8)
    hsv[..., 2] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = (np.clip(mag / 20, 0, 1) * 255).astype('uint8')
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb
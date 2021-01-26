'''
All neural networks and layers are defined here.
The main functions here are: PoseNet and DepthNet.
'''

from load_config import *

def get_pretrained_mobilenetv2(n_inputs=1, name=None, n_downsamples=4):
    # Make model with desired amount of inputs
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[config.image_h, config.image_w, 3 * n_inputs],
        include_top=False,
        weights=None,
    )
    # base_model.summary()
    # Load pretrained model
    base_model_trained = tf.keras.applications.MobileNetV2(
        input_shape=[config.image_h, config.image_w, 3],
        include_top=False,
        weights='imagenet',
    )
    # Copy weights
    stop_pretrained = False
    for v, v_trained in zip(
        base_model.trainable_variables,
        base_model_trained.trainable_variables,
    ):
        if False: #re.match('block_13(.+)', v.name) or stop_pretrained:
            print('Randomly initializing layer', v.name)
            stop_pretrained = True
        if re.match('Conv1(.+)kernel:0', v.name):
            new_kernel = tf.concat([v_trained] * n_inputs, 2) / n_inputs
            new_kernel = new_kernel.numpy()
            noise = np.random.random(size=new_kernel.shape) * new_kernel.std() * config.init_binocular_noise
            scale = 1 / np.sqrt(1 + config.init_binocular_noise**2)
            v.assign((new_kernel + noise) * scale)
            print('kernel found:', v.name)
        else:
            v.assign(v_trained)
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # /2
        'block_3_expand_relu',   # /4
        'block_6_expand_relu',   # /8
        'block_13_expand_relu',  # /16
        #'block_16_project',      # /32
        'out_relu',              # /32
    ][:n_downsamples]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers, name=name)
    
    # Scale image between -1 and 1
    inputs = kl.Input(shape=[config.image_h, config.image_w, 3 * n_inputs])
    x = inputs * 2 - 1
    # x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255)
    outputs = down_stack(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    
    return model

def get_pretrained_resnet50(n_inputs=1, name=None):
    # Make model with desired amount of inputs
    base_model = tf.keras.applications.ResNet50(
        input_shape=[config.image_h, config.image_w, 3 * n_inputs],
        include_top=False,
        weights=None,
    )
    # base_model.summary()
    # Load pretrained model
    base_model_trained = tf.keras.applications.ResNet50(
        input_shape=[config.image_h, config.image_w, 3],
        include_top=False,
        weights='imagenet',
    )
    # Copy weights
    for v, v_trained in zip(
        base_model.trainable_variables,
        base_model_trained.trainable_variables,
    ):
        if re.match('conv1_conv(.+)kernel:0', v.name):
            new_kernel = tf.concat([v_trained] * n_inputs, 2) / n_inputs
            new_kernel = new_kernel.numpy()
            noise = np.random.random(size=new_kernel.shape) * new_kernel.std() * config.init_binocular_noise
            scale = 1 / np.sqrt(1 + config.init_binocular_noise**2)
            v.assign((new_kernel + noise) * scale)
            print('kernel found:', v.name)
        else:
            v.assign(v_trained)
    # Use the activations of these layers
    layer_names = [
        'conv1_relu',         # /2
        'conv2_block3_out',   # /4
        'conv3_block4_out',   # /8
        'conv4_block6_out',   # /16
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers, name=name)
    return down_stack

def get_pretrained_vgg16(name=None):
    # Load pretrained model
    base_model = tf.keras.applications.VGG16(
        input_shape=[config.image_h, config.image_w, 3],
        include_top=False,
        weights='imagenet',
    )
    # base_model.summary()
    # Use the activations of these layers
    layer_name = 'block1_conv2'
    layer = base_model.get_layer(layer_name).output

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layer, name=name)
    down_stack.trainable = False

    # Preprocess image
    inputs = kl.Input(shape=[config.image_h, config.image_w, 3])
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    outputs = down_stack(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model

def get_pretrained(backbone, n_inputs=1, name=None, *args, **kwargs):
    backbone_fn = {
        'resnet50': get_pretrained_resnet50,
        'mobilenetv2': get_pretrained_mobilenetv2,
    }[backbone]
    backbone = backbone_fn(n_inputs=n_inputs, name=name, *args, **kwargs)
    return backbone

def reflection_pad(x):
    # Reflection padding
    x = tf.concat([x[:, 1:2], x, x[:, -2:-1]], 1)
    x = tf.concat([x[:, :, 1:2], x, x[:, :, -2:-1]], 2)
    return x

class BasicBlock(kl.Layer):
    def __init__(self, n_units, stride=1, downsample=None, activation='relu'):
        super().__init__()
        self.n_units = n_units
        self.stride = stride
        self.downsample = downsample
        self.activation = activation

        self.conv1 = kl.Conv2D(
            n_units, 3,
            strides=(stride, stride),
            padding='valid',
            use_bias=False)
        self.conv2 = kl.Conv2D(
            n_units, 3,
            padding='valid',
            use_bias=False)
        self.bn1 = kl.BatchNormalization(axis=-1)
        self.bn2 = kl.BatchNormalization(axis=-1)
        self.downsample = keras.Sequential([
            kl.Conv2D(n_units, 1, strides=(stride, stride), padding='valid'),
            kl.BatchNormalization(axis=-1),
        ])
        if activation == 'relu':
            self.activation = keras.activations.relu
        elif activation is None:
            self.activation = tf.identity
    
    def call(self, x):
        identity = x
        
        x = reflection_pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = keras.activations.relu(x)
        
        x = reflection_pad(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = (x + self.downsample(identity)) * np.sqrt(0.5)
        x = self.activation(x)
        
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_units': self.n_units,
            'stride': self.stride,
            'downsample': self.downsample,
            'activation': self.activation,
        })
        return config

class UpSamplingBlock(kl.Layer):
    def __init__(self, n_units, name=None):
        super().__init__(name=name)
        self.up = kl.UpSampling2D()
        self.n_units = n_units
        self.conv = BasicBlock(n_units)
    
    def call(self, x, residual=None):
        x = self.up(x)
        if residual is not None:
            x = tf.concat([x, residual], 3)
        return self.conv(x)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_units': self.n_units,
        })
        return config


def FlowNet():
    #
    # Define Layers
    #
    
    # Encoder
    backbone = get_pretrained(config.backbone_flow, n_inputs=2, name='backbone')

    # Flow decoder
    upsample_16_8 = UpSamplingBlock(256, name='upsample_16_8')
    upsample_8_4 = UpSamplingBlock(128, name='upsample_8_4')
    upsample_4_2 = UpSamplingBlock(64, name='upsample_4_2')
    upsample_2_1 = UpSamplingBlock(32, name='upsample_2_1')

    # Final convolution layers
    final_conv_1 = kl.Conv2D(2, 1, padding='same', activation=None, name='final_conv_1')
    final_conv_mask = kl.Conv2D(1, 1, padding='same', activation='sigmoid', name='final_conv_mask')
    final_conv_2 = keras.Sequential([
            kl.Conv2D(2, 1, padding='same', activation=None),
            kl.UpSampling2D(),
        ], name='final_conv_2')
    final_conv_4 = keras.Sequential([
            kl.Conv2D(2, 1, padding='same', activation=None),
            kl.UpSampling2D(),
            kl.UpSampling2D(),
        ], name='final_conv_4')
    final_conv_8 = keras.Sequential([
            kl.Conv2D(2, 1, padding='same', activation=None),
            kl.UpSampling2D(),
            kl.UpSampling2D(),
            kl.UpSampling2D(),
        ], name='final_conv_8')
    
    #
    # Define graph
    #
    
    # Convolutions
    inputs = kl.Input(shape=[config.image_h, config.image_w, 6])
    x2, x4, x8, x16 = backbone(inputs)

    # Upsamplings
    x = x16
    x = upsample_16_8(x, x8)
    flow_8 = final_conv_8(x)

    x = upsample_8_4(x, x4)
    flow_4 = final_conv_4(x)

    x = upsample_4_2(x, x2)
    flow_2 = final_conv_2(x)

    x = upsample_2_1(x)
    flow_1 = final_conv_1(x)
    flow_mask = final_conv_mask(x*0.01)

    return Model(inputs=inputs, outputs=[flow_1, flow_2, flow_4, flow_8, flow_mask], name='flow_net')


def DepthNet():
    #
    # Define Layers
    #
    
    # Encoder
    backbone = get_pretrained(config.backbone_depth, n_inputs=2, name='backbone')

    # Depth decoder
    upsample_16_8 = UpSamplingBlock(256, name='upsample_16_8')
    upsample_8_4 = UpSamplingBlock(128, name='upsample_8_4')
    upsample_4_2 = UpSamplingBlock(64, name='upsample_4_2')
    upsample_2_1 = UpSamplingBlock(32, name='upsample_2_1')

    # Final convolution layers
    final_conv_1 = kl.Conv2D(1, 1, padding='same', activation=None, name='final_conv_1')
    final_conv_2 = keras.Sequential([
            kl.Conv2D(1, 1, padding='same', activation=None),
            kl.UpSampling2D(),
        ], name='final_conv_2')
    final_conv_4 = keras.Sequential([
            kl.Conv2D(1, 1, padding='same', activation=None),
            kl.UpSampling2D(),
            kl.UpSampling2D(),
        ], name='final_conv_4')
    final_conv_8 = keras.Sequential([
            kl.Conv2D(1, 1, padding='same', activation=None),
            kl.UpSampling2D(),
            kl.UpSampling2D(),
            kl.UpSampling2D(),
        ], name='final_conv_8')
    
    #
    # Define graph
    #
    
    # Convolutions
    inputs = kl.Input(shape=[config.image_h, config.image_w, 6])
    x2, x4, x8, x16 = backbone(inputs)

    # Upsamplings
    x = x16
    x = upsample_16_8(x, x8)
    depth_8 = final_conv_8(x)

    x = upsample_8_4(x, x4)
    depth_4 = final_conv_4(x)

    x = upsample_4_2(x, x2)
    depth_2 = final_conv_2(x)

    x = upsample_2_1(x)
    depth_1 = final_conv_1(x)

    # Depth prediction
    depths = []
    for depth in [depth_1, depth_2, depth_4, depth_8]:
        # formula from https://arxiv.org/pdf/1806.01260.pdf (monodepth2)
        a = 1 / config.min_depth
        b = 1 / config.max_depth
        depth = 1 / (a * tf.math.sigmoid(depth - 2) + b)
        depths.append(depth[:, :, :, 0])

    return Model(inputs=inputs, outputs=depths, name='depth_net')

def PoseNet():
    #
    # Define Layers
    #
    
    # Encoder
    backbone = get_pretrained(config.backbone_pose, n_inputs=4, name='backbone', n_downsamples=5)

    flatten = kl.GlobalAveragePooling2D()
    rotation_net = keras.Sequential([
        kl.Dense(512, activation=config.posenet_activation),
        kl.Dense(512, activation=config.posenet_activation),
        kl.Dense(3, activation=None),
    ], name='rotation_net')
    translation_net = keras.Sequential([
        kl.Dense(512, activation=config.posenet_activation),
        kl.Dense(512, activation=config.posenet_activation),
        kl.Dense(3, activation=None),
    ], name='translation_net')
    concat = kl.Concatenate(axis=1)
    
    #
    # Define graph
    #
    
    # Convolutions
    inputs = kl.Input(shape=[config.image_h, config.image_w, 12])
    x2, x4, x8, x16, x32 = backbone(inputs)

    # Flatten
    x = flatten(x32)
    # Predict pose
    rotation = rotation_net(x) * config.scale_rot # [batch, 3]
    translation = translation_net(x) * config.scale_tr # [batch, 3]
    output = concat([rotation, translation]) # [batch, 6]
    
    return Model(inputs=inputs, outputs=output, name='pose_net')
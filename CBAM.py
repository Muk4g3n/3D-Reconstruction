import tensorflow as tf
from tensorflow.keras import layers,losses
class ConvolutionBlockAttentionModule(layers.Layer):
    def __init__(self, reduction_ratio=16, use_global_pooling=True):
        super(ConvolutionBlockAttentionModule, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.use_global_pooling = use_global_pooling
        self.conv1 = layers.Conv2D(filters=int(reduction_ratio/2), kernel_size=1, strides=1, padding='same')
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters=int(reduction_ratio/2), kernel_size=1, strides=1, padding='same')
        self.relu2 = layers.ReLU()
        self.conv3 = layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same')
        self.sigmoid = layers.Activation('sigmoid')
        if use_global_pooling:
            self.global_pooling = layers.GlobalAveragePooling2D()

    def call(self, inputs):
        if self.use_global_pooling:
            channel_att_weights = self.global_pooling(inputs)
            channel_att_weights = tf.expand_dims(tf.expand_dims(channel_att_weights, axis=1), axis=1)
        else:
            channel_att_weights = inputs

        channel_att_weights = self.conv1(channel_att_weights)
        channel_att_weights = self.relu1(channel_att_weights)
        channel_att_weights = self.conv2(channel_att_weights)
        channel_att_weights = self.relu2(channel_att_weights)
        channel_att_weights = self.conv3(channel_att_weights)
        channel_att_weights = self.sigmoid(channel_att_weights)

        spatial_att_weights = 1 - channel_att_weights
        output = inputs * channel_att_weights + inputs * spatial_att_weights

        return output
import tensorflow as tf
import tensorflow.nn as nn
import tensorflow.keras.layers as layers


class Convolution(tf.keras.Model):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9, strides=1, padding='valid'):
        super(Convolution, self).__init__()
        self.conv = layers.conv2d(
            inputs=in_channels,
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding
        )
        self.relu = nn.relu()

    def call(self, x):
        out_conv = self.conv(x)
        out_relu = self.relu(out_conv)
        return out_relu


class PrimaryCaps(tf.keras.Model):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, padding='valid'):
        super(PrimaryCaps, self).__init__()
        self.capsules = []

    def call(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = tf.stack(u, axis=1)
        u = tf.reshape(u, [x.size(0), 32 * 6 * 6 * -1])
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = tf.math.reduce_sum((input_tensor**2), axis=-1, keepdims=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * tf.math.sqrt(squared_norm))
        return output_tensor


class DigitCaps(tf.keras.Model):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_capsules = num_capsules

        self.W = tf.Variable(tf.random.normal(shape=(1, num_routes, num_capsules, out_channels, in_channels)))

    def call(self, x):
        return NotImplementedError

    
    def squash(self, input_tensor):
        squared_norm = tf.math.reduce_sum((input_tensor**2), axis=-1, keepdims=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * tf.math.sqrt(squared_norm))
        return output_tensor


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.reconstraction_layers = tf.keras.Sequential([
            layers.Dense(inputs=16 * 10, units=512, activation=nn.relu),
            layers.Dense(inputs=512, units=1024, activation=nn.relu),
            layers.Dense(input=1024, units=784, activation=tf.keras.activations.sigmoid)
        ])

    def call(self, x):
        return NotImplementedError
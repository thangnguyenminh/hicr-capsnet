import tensorflow as tf
import layers
import tf.nn as F


class CapsNet(tf.keras.Model):
    """Capsule Network Model"""
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv_layer = layers.Convolution()
        self.primary_caps = layers.PrimaryCaps()
        self.digit_caps = layers.DigitCaps()
        self.decoder = layers.Decoder()

    def call(self, x):
        output = self.digit_caps(self.primary_caps(self.conv_layer(x)))
        reconstructions, masked = self.decoder(output, x)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = tf.math.sqrt(tf.reduce_sum(x**2, axis=2, keepdims=True))

        left = tf.reshape(F.relu(0.9 - v_c), (batch_size, -1))
        right = tf.reshape(F.relu(v_c - 0.1), (batch_size, -1))

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = tf.metrics.mean(tf.reduce_sum(loss, axis=1))

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = tf.losses.mean_squared_error(reconstructions.reshape(reconstructions.size(0), -1), data.reshape(reconstructions.size(0), -1))
        return loss * 0.0005
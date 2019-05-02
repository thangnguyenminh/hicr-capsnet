import keras.backend as K
import tensorflow as tf
from keras import initializers, layers


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    Args:
        vectors (vectors): some vectors to be squashed, N-dim tensor
        axis (int): the axis to squash
    Returns:
        product (tensor): Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

def PrimaryCaps(inputs, output_channels, kernel_size, strides, padding, output_capsule_dim, name):
    """ 
    Apply Conv2D `output_channels` times and concatenate all capsules  
    Args:
        inputs (tensor): 4D tensor, shape=[None, width, height, channels]
        output_capsule_dim (int): the dimension of the output vector of capsule
        output_channels (int): the number of types of capsules
    Returns:
        output (tensor): Output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filters=output_capsule_dim*output_channels, kernel_size=kernel_size, strides=strides, padding=padding, name=name)(inputs)
    outputs = layers.Reshape(target_shape=[-1, output_capsule_dim], name='primary_capsule_reshaped')(output)
    return layers.Lambda(squash, name='primary_capsule_squashed')(outputs)

class DigitCaps(layers.Layer):
    """
    The digit capsule layer. It is similar to Dense layer. 
    Digit Capsule Layer just expand the output of the neuron from scalar to vector
    Args:
        routings (int): number of iterations for the routing algorithm
        output_capsule_dim (int): dimension of the output vectors of the capsules in this layer
        output_channels (int): number of capsules in this layer
    Returns:
    """
    def __init__(self, output_channels, output_capsule_dim, routings=3, kernel_initializer='glorot_uniform', **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.output_channels = output_channels
        self.output_capsule_dim = output_capsule_dim
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_channels = input_shape[1]
        self.input_capsule_dim = input_shape[2]
        
        self.W = self.add_weight(shape=[self.output_channels, self.input_channels,
                                        self.output_capsule_dim, self.input_capsule_dim],
                                 initializer=self.kernel_initializer,
                                 name='W')
        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(inputs, 1)
        inputs_tiled = K.tile(inputs_expand, [1, self.output_channels, 1, 1])
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Routing Algorithm Starts
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.output_channels, self.input_channels])
        assert self.routings > 0, 'The routings should be greater than 0'
        for i in range(self.routings):
            c = tf.nn.softmax(b, dim=1)
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))
            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # Routing Algorithm Ends

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.output_channels, self.output_capsule_dim])

    def get_config(self):
        config = {
            'output_channels': self.output_channels,
            'output_capsule_dim': self.output_capsule_dim,
            'routings': self.routings
        }
        base_config = super(DigitCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    Args:
        inputs (tensor): shape=[None, num_vectors, dim_vector]
    Returns:
        output (tensor): shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config

class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config
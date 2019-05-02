import numpy as np
from keras import backend as K
from keras import models, optimizers
from keras.layers import Conv2D, Dense, Input, MaxPooling2D, Reshape
from keras.models import Sequential

from base.base_model import BaseModel
from models.capsnet_layers import DigitCaps, Length, Mask, PrimaryCaps
from models.loss import margin_loss

K.set_image_data_format('channels_last')

class CapsNetModel(BaseModel):
    """ Capsule Network with Dynamic Routing for HICR """
    def __init__(self, config):
        super(CapsNetModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        """ Builds the model layer by layer """
        # From Paper: Win x Hin x Cin (Width x Height x Input Channels)
        x = Input(shape=self.config.model.input_shape)
        
        # From Paper: Typical Convolution Layer which converts the input image into a block of activations
        # TYPE OF LAYER = Convolution, KERNEL SIZE = 9, STRIDE = 1, PADDING = 0, INPUT CHANNELS = 1, OUTPUT CHANNELS = 256 
        convolution = Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='convolution')(x)
        
        # From Paper: Primary Capsule Layer  passing to the next layer this tensor is reshaped into N pc × D pc .
        # TYPE OF LAYER = Primary Caps, KERNEL SIZE = 9, STRIDE = 2, PADDING = 0, INPUT CHANNELS = 256, OUTPUT CHANNELS = 32,
        # INPUT CAPSULE DIMENSION = 1, OUTPUT CAPSULE DIMENSION = 8
        primarycaps = PrimaryCaps(convolution, output_channels=32, kernel_size=9, strides=2, padding='valid', output_capsule_dim=8, name='primary_capsule')
        
        # From Paper: Digit Capsule Layer is the fully connected network of the shape Nclass × 1
        # TYPE OF LAYER = Digit Caps, KERNEL SIZE = NA, STRIDE = NA, PADDING = NA, INPUT CHANNELS = 1152, OUTPUT CHANNELS = 10
        # INPUT CAPSULE DIMENSION = 8, OUTPUT CAPSULE DIMENSION = 16
        digitcaps = DigitCaps(routings=self.config.model.routings, output_channels=10, output_capsule_dim=16, name='digit_capsule')(primarycaps)
        
        # Output from Capsule Layers to replace each capsule with its length
        capsule_output = Length(name='capsule_output')(digitcaps)

        # From Paper: Decoder Network
        y = Input(shape=(self.config.model.num_classes,))
        # The true label is used to mask the output of capsule layer. For training
        masked_by_y = Mask()([digitcaps, y])  
        # TYPE OF LAYERS = Decoder FC, KERNEL SIZE = STRIDE = PADDING = NA, INPUT CHANNELS = 160, OUTPUT CHANNELS = 512
        decoder = Sequential(name='decoder')
        # Here num_classes = 10 x 16 = 160
        decoder.add(Dense(512, activation='relu', input_dim=16*self.config.model.num_classes))
        # TYPE OF LAYERS = Decoder FC, KERNEL SIZE = STRIDE = PADDING = NA, INPUT CHANNELS = 512, OUTPUT CHANNELS = 1024
        decoder.add(Dense(1024, activation='relu'))
        # TYPE OF LAYERS = Decoder FC, KERNEL SIZE = STRIDE = PADDING = NA, INPUT CHANNELS = 1024, OUTPUT CHANNELS = 784
        # here input_shape = (28, 28) :: 28 x 28 = 784
        decoder.add(Dense(np.prod(self.config.model.input_shape), activation='sigmoid'))
        decoder.add(Reshape(target_shape=self.config.model.input_shape, name='reconstructed_output'))

        self.model = models.Model([x, y], [capsule_output, decoder(masked_by_y)])

        self.model.compile(
            optimizer=optimizers.Adam(lr=self.config.model.learning_rate),
            loss=[margin_loss, 'mse'],
            loss_weights=[1., self.config.model.lam_recon],
            metrics=['acc']
        )

        self.model.summary()
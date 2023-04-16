import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.callbacks import ReduceLROnPlateau
from tensorflow import keras
from keras import layers
from localization_losses import naive_loss
from localization import Localization


class FaceLocalizerAndRecognizer:
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-8)
        self.localization_loss_function = naive_loss
        self.classification_loss_function = tf.keras.losses.BinaryCrossentropy()
        self.learning_rate_reducer = ReduceLROnPlateau(monitor='total_loss', factor=0.1, patience=10, verbose=1,
                                                       mode='min', min_lr=1e-7)
        self.callbacks = [
            keras.callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=1,
                embeddings_freq=1
            )
        ]
        self.neural_network = None

    def build_neural_network(self, width, height, channel):
        conv_base = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=(width, height, channel))

        conv_base.trainable = True
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        input_layer = layers.Input(shape=(width, height, channel))
        vgg = conv_base(input_layer)
        x = layers.Flatten()(vgg)
        x = layers.Dropout(0.6)(x)
        x = layers.Dense(units=256, activation='relu')(x)
        x = layers.Dropout(0.6)(x)
        classification_output = layers.Dense(units=1, activation='sigmoid')(x)

        y = layers.Flatten()(vgg)
        y = layers.Dropout(0.6)(y)
        y = layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))(y)
        y = layers.Dropout(0.6)(y)
        regression_output = layers.Dense(4, activation='sigmoid')(y)

        self.neural_network = Model(inputs=input_layer, outputs=[classification_output, regression_output])

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_localization_loss_function(self, loss_function):
        self.localization_loss_function = loss_function

    def set_classification_loss_function(self, loss_function):
        self.classification_loss_function = loss_function

    def save_model(self, filename):
        self.neural_network.save(filename)

    def build_model(self):
        self.build_neural_network(224, 224, 3)
        localization_model = Localization(
            self.neural_network,
            self.optimizer,
            self.classification_loss_function,
            self.localization_loss_function
        )
        localization_model.compile()
        return localization_model


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, Dense
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, MaxPooling3D, TimeDistributed, Flatten
from tensorflow.keras.regularizers import l2

def build_model(input_shape=(16,224,224,3), num_classes=1, kernel_initializer="he_normal", reg_factor=5e-4, drop_rate=None):
    kernel_regularizer = l2(reg_factor)
    inp = Input(shape=input_shape)
    x = Conv3D(16, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='same',
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 2, 2), padding="same")(x)

    x = Conv3D(32, kernel_size=(1, 3, 3), strides=(1, 2, 2), padding='same',
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(32, kernel_size=(3, 1, 1), strides=(1, 1, 1), padding='same',
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(64, kernel_size=(1, 3, 3), strides=(1, 2, 2), padding='same',
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(64, kernel_size=(3, 1, 1), strides=(2, 1, 1), padding='same',
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(128, kernel_size=(1, 3, 3), strides=(1, 2, 2), padding='same',
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(128, kernel_size=(3, 1, 1), strides=(2, 1, 1), padding='same',
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = TimeDistributed(GlobalAveragePooling2D())(x)
    embedded_vector = Flatten()(x)
    if drop_rate is not None:
        embedded_vector = Dropout(drop_rate)(embedded_vector)
    out = Dense(num_classes, activation='sigmoid', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(embedded_vector)
    model = Model(inputs=inp, outputs=[out, embedded_vector])
    return model

# model = build_model()
# model.summary(line_length=150)
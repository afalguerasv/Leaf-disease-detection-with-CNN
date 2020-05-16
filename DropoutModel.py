import tensorflow as tf
from keras import regularizers

def dropoutModel():
    with tf.device('/gpu:0'):
        DropoutModel = tf.keras.models.Sequential()
        DropoutModel.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
        # DropoutModel.add(tf.keras.layers.BatchNormalization())
        DropoutModel.add(tf.keras.layers.Activation('relu'))
        DropoutModel.add(tf.keras.layers.Dropout(0.05))
        DropoutModel.add(tf.keras.layers.MaxPooling2D(2, 2))

        DropoutModel.add(tf.keras.layers.Conv2D(64, (3, 3)))
        # DropoutModel.add(tf.keras.layers.BatchNormalization())
        DropoutModel.add(tf.keras.layers.Activation('relu'))
        DropoutModel.add(tf.keras.layers.Dropout(0.05))
        DropoutModel.add(tf.keras.layers.MaxPooling2D(2, 2))

        DropoutModel.add(tf.keras.layers.Conv2D(128, (3, 3)))
        # DropoutModel.add(tf.keras.layers.BatchNormalization())
        DropoutModel.add(tf.keras.layers.Activation('relu'))
        DropoutModel.add(tf.keras.layers.Dropout(0.05))
        DropoutModel.add(tf.keras.layers.MaxPooling2D(2, 2))


        DropoutModel.add(tf.keras.layers.Conv2D(128, (3, 3)))
        # DropoutModel.add(tf.keras.layers.BatchNormalization())
        DropoutModel.add(tf.keras.layers.Activation('relu'))
        DropoutModel.add(tf.keras.layers.Dropout(0.05))
        DropoutModel.add(tf.keras.layers.MaxPooling2D(2, 2))

        DropoutModel.add(tf.keras.layers.Flatten())
        DropoutModel.add(tf.keras.layers.Dense(512))
        # DropoutModel.add(tf.keras.layers.BatchNormalization())
        DropoutModel.add(tf.keras.layers.Activation('relu'))
        DropoutModel.add(tf.keras.layers.Dropout(0.05))
        DropoutModel.add(tf.keras.layers.Dense(16))

        return DropoutModel
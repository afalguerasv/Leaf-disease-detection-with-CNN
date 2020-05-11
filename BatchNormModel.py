import tensorflow as tf

def batchNormModel():
    batchNormModel = tf.keras.models.Sequential()

    batchNormModel.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
    batchNormModel.add(tf.keras.layers.BatchNormalization())
    batchNormModel.add(tf.keras.layers.Activation('relu'))
    batchNormModel.add(tf.keras.layers.MaxPooling2D(2, 2))

    batchNormModel.add(tf.keras.layers.Conv2D(64, (3, 3)))
    batchNormModel.add(tf.keras.layers.BatchNormalization())
    batchNormModel.add(tf.keras.layers.Activation('relu'))
    batchNormModel.add(tf.keras.layers.MaxPooling2D(2, 2))

    batchNormModel.add(tf.keras.layers.Conv2D(128, (3, 3)))
    batchNormModel.add(tf.keras.layers.BatchNormalization())
    batchNormModel.add(tf.keras.layers.Activation('relu'))
    batchNormModel.add(tf.keras.layers.MaxPooling2D(2, 2))

    batchNormModel.add(tf.keras.layers.Conv2D(128, (3, 3)))
    batchNormModel.add(tf.keras.layers.BatchNormalization())
    batchNormModel.add(tf.keras.layers.Activation('relu'))
    batchNormModel.add(tf.keras.layers.MaxPooling2D(2, 2))

    batchNormModel.add(tf.keras.layers.Flatten())
    batchNormModel.add(tf.keras.layers.Dense(512))
    batchNormModel.add(tf.keras.layers.BatchNormalization())
    batchNormModel.add(tf.keras.layers.Activation('relu'))
    batchNormModel.add(tf.keras.layers.Dense(16))
    batchNormModel.add(tf.keras.layers.Dropout(0.2))

    return batchNormModel
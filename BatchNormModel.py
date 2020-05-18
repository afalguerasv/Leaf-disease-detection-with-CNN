import tensorflow as tf

def batchNormModel():
    batchNormModel = tf.keras.models.Sequential()

    batchNormModel.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    batchNormModel.add(tf.keras.layers.BatchNormalization(axis=1))
    batchNormModel.add(tf.keras.layers.MaxPooling2D(2, 2))

    batchNormModel.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    batchNormModel.add(tf.keras.layers.BatchNormalization(axis=1))
    batchNormModel.add(tf.keras.layers.MaxPooling2D(2, 2))

    batchNormModel.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    batchNormModel.add(tf.keras.layers.BatchNormalization(axis=1))
    batchNormModel.add(tf.keras.layers.MaxPooling2D(2, 2))

    batchNormModel.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    batchNormModel.add(tf.keras.layers.BatchNormalization(axis=1))
    batchNormModel.add(tf.keras.layers.MaxPooling2D(2, 2))

    batchNormModel.add(tf.keras.layers.Flatten())
    batchNormModel.add(tf.keras.layers.Dense(512, activation='relu'))
    batchNormModel.add(tf.keras.layers.BatchNormalization(axis=1))
    batchNormModel.add(tf.keras.layers.Dense(16))

    return batchNormModel
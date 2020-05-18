import tensorflow as tf

def L2Model():
    with tf.device('/gpu:0'):
        l2Model = tf.keras.models.Sequential()
        l2Model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
        # l2Model.add(tf.keras.layers.BatchNormalization())
        l2Model.add(tf.keras.layers.Activation('relu'))
        l2Model.add(tf.keras.layers.ActivityRegularization(l2=0.0001))
        l2Model.add(tf.keras.layers.MaxPooling2D(2, 2))

        l2Model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        # l2Model.add(tf.keras.layers.BatchNormalization())
        l2Model.add(tf.keras.layers.Activation('relu'))
        l2Model.add(tf.keras.layers.ActivityRegularization(l2=0.0001))
        l2Model.add(tf.keras.layers.MaxPooling2D(2, 2))

        l2Model.add(tf.keras.layers.Conv2D(128, (3, 3)))
        # l2Model.add(tf.keras.layers.BatchNormalization())
        l2Model.add(tf.keras.layers.Activation('relu'))
        l2Model.add(tf.keras.layers.ActivityRegularization(l2=0.0001))
        l2Model.add(tf.keras.layers.MaxPooling2D(2, 2))


        l2Model.add(tf.keras.layers.Conv2D(128, (3, 3)))
        # l2Model.add(tf.keras.layers.BatchNormalization())
        l2Model.add(tf.keras.layers.Activation('relu'))
        l2Model.add(tf.keras.layers.ActivityRegularization(l2=0.0001))
        l2Model.add(tf.keras.layers.MaxPooling2D(2, 2))

        l2Model.add(tf.keras.layers.Flatten())
        l2Model.add(tf.keras.layers.Dense(512))
        # l2Model.add(tf.keras.layers.BatchNormalization())
        l2Model.add(tf.keras.layers.Activation('relu'))
        l2Model.add(tf.keras.layers.Dense(16))

        return l2Model
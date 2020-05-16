from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import datetime


logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print(tf.version.VERSION, tf.executing_eagerly(), tf.keras.layers.BatchNormalization._USE_V2_BEHAVIOR)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

base_dir = os.path.dirname('B:\TFG\PlantVillage2\\')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

EPOCHS = 20
BATCH_SIZE = 64
IMG_SHAPE = 100

def plotImages(images_arr):
    # plot array of images with 1 row and 5 columns
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

total_train_images = 0  # total files
total_validation_images = 0
for dirpath, dirnames, filenames in os.walk(train_dir):
    N_c = len(filenames)
    total_train_images += N_c
    print("Files in ", dirpath, N_c)
print("Total Train Files ",total_train_images)

for dirpath, dirnames, filenames in os.walk(val_dir):
    N_c = len(filenames)
    total_validation_images += N_c
    print("Files in ", dirpath, N_c)
print("Total Validation Files ",total_validation_images)

image_gen_train = ImageDataGenerator(rescale=1./255)


train_data_gen = image_gen_train.flow_from_directory(target_size=(100, 100),
                                                     batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     color_mode='rgb',
                                                     shuffle=True,
                                                     class_mode='categorical')

# print('test data gen: ', test_data_gen[0][0])
# plotImages(test_data_gen[0][0])

# example of how looks single image five times with random augmentations
# augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(target_size=(100, 100),
                                                 batch_size=BATCH_SIZE,
                                                 directory=val_dir,
                                                 class_mode='categorical')

print(val_data_gen)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(16)
])
#LEARNING_RATE = 0.001
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

#Log directory
log_dir = ".\\logs\\test\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "Simple"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


print(type(np.ceil(16888 / float(BATCH_SIZE))))


history = model.fit_generator(train_data_gen,
                              epochs=EPOCHS,
                              steps_per_epoch=int(np.ceil(total_train_images / float(BATCH_SIZE))),
                              validation_data=val_data_gen,
                              validation_steps=int(np.ceil(total_validation_images / float(BATCH_SIZE))),
                              verbose=2,
                              callbacks=[tensorboard_callback])

model.save('.\\models\\SimpleModel\\simpleModel.h5')
acc = history.history['accuracy']
print('acc ', acc)
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)
print('epochs range: ', epochs_range)

plt.figure(figsize=(8,8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

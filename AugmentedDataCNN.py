from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import datetime
import BatchNormModel as BNM

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# print(tf.version.VERSION, tf.executing_eagerly(), tf.keras.layers.BatchNormalization._USE_V2_BEHAVIOR)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

base_dir = os.path.dirname('B:\TFG\PlantVillage\\')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
EPOCHS = 10

# Path vars for TRAIN sets
train_dir_Cherry_healthy = os.path.join(train_dir, 'Cherry___healthy')
train_dir_Cherry_Powdery_mildew = os.path.join(train_dir, 'Cherry___Powdery_mildew')
train_dir_Grape_healthy = os.path.join(train_dir, 'Grape___healthy')
train_dir_Grape_Black_rot = os.path.join(train_dir, 'Grape___Black_rot')
train_dir_Grape_Esca_Black_Measles = os.path.join(train_dir, 'Grape___Esca_Black_Measles')
train_dir_Grape_Leaf_blight = os.path.join(train_dir, 'Grape___Leaf_blight_Isariopsis_Leaf_Spot')
train_dir_Tomato_healthy = os.path.join(train_dir, 'Tomato___healthy')
train_dir_Tomato_Bacterial_Spot = os.path.join(train_dir, 'Tomato___Bacterial_spot')
train_dir_Tomato_Early_blight = os.path.join(train_dir, 'Tomato___Early_blight')
train_dir_Tomato_Late_blight = os.path.join(train_dir, 'Tomato___Late_blight')
train_dir_Tomato_Leaf_Mold = os.path.join(train_dir, 'Tomato___Leaf_Mold')
train_dir_Tomato_Septoria_leaf_spot = os.path.join(train_dir, 'Tomato___Septoria_leaf_spot')
train_dir_Tomato_Spider_mites = os.path.join(train_dir, 'Tomato___Spider_mites_Two-spotted_spider_mite')
train_dir_Tomato_Target_Spot = os.path.join(train_dir, 'Tomato___Target_Spot')
train_dir_Tomato_mosaic_virus = os.path.join(train_dir, 'Tomato___Tomato_mosaic_virus')
train_dir_Tomato_Yellow_Leaf_Curl_Virus = os.path.join(train_dir, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus')

# Path vars for VALIDATION sets
val_dir_Cherry_healthy = os.path.join(val_dir, 'Cherry___healthy')
val_dir_Cherry_Powdery_mildew = os.path.join(val_dir, 'Cherry___Powdery_mildew')
val_dir_Grape_healthy = os.path.join(val_dir, 'Grape___healthy')
val_dir_Grape_Black_rot = os.path.join(val_dir, 'Grape___Black_rot')
val_dir_Grape_Esca_Black_Measles = os.path.join(val_dir, 'Grape___Esca_Black_Measles')
val_dir_Grape_Leaf_blight = os.path.join(val_dir, 'Grape___Leaf_blight_Isariopsis_Leaf_Spot')
val_dir_Tomato_healthy = os.path.join(val_dir, 'Tomato___healthy')
val_dir_Tomato_Bacterial_Spot = os.path.join(val_dir, 'Tomato___Bacterial_spot')
val_dir_Tomato_Early_blight = os.path.join(val_dir, 'Tomato___Early_blight')
val_dir_Tomato_Late_blight = os.path.join(val_dir, 'Tomato___Late_blight')
val_dir_Tomato_Leaf_Mold = os.path.join(val_dir, 'Tomato___Leaf_Mold')
val_dir_Tomato_Septoria_leaf_spot = os.path.join(val_dir, 'Tomato___Septoria_leaf_spot')
val_dir_Tomato_Spider_mites = os.path.join(val_dir, 'Tomato___Spider_mites_Two-spotted_spider_mite')
val_dir_Tomato_Target_Spot = os.path.join(val_dir, 'Tomato___Target_Spot')
val_dir_Tomato_mosaic_virus = os.path.join(val_dir, 'Tomato___Tomato_mosaic_virus')
val_dir_Tomato_Yellow_Leaf_Curl_Virus = os.path.join(val_dir, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus')

# Let's count how many train and validation images we have
num_train_Cherry_healthy = len(os.listdir(train_dir_Cherry_healthy))
num_train_Cherry_Powdery_mildew = len(os.listdir(train_dir_Cherry_Powdery_mildew))
num_train_Grape_healthy = len(os.listdir(train_dir_Grape_healthy))
num_train_Grape_Black_rot = len(os.listdir(train_dir_Grape_Black_rot))
num_train_Grape_Esca_Black_Measles = len(os.listdir(train_dir_Grape_Esca_Black_Measles))
num_train_Grape_Leaf_blight = len(os.listdir(train_dir_Grape_Leaf_blight))
num_train_Tomato_healthy = len(os.listdir(train_dir_Tomato_healthy))
num_train_Tomato_Bacterial_Spot = len(os.listdir(train_dir_Tomato_Bacterial_Spot))
num_train_Tomato_Early_blight = len(os.listdir(train_dir_Tomato_Early_blight))
num_train_Tomato_Late_blight = len(os.listdir(train_dir_Tomato_Late_blight))
num_train_Tomato_Leaf_Mold = len(os.listdir(train_dir_Tomato_Leaf_Mold))
num_train_Tomato_Septoria_leaf_spot = len(os.listdir(train_dir_Tomato_Septoria_leaf_spot))
num_train_Tomato_Spider_mites = len(os.listdir(train_dir_Tomato_Spider_mites))
num_train_Tomato_Target_Spot = len(os.listdir(train_dir_Tomato_Target_Spot))
num_train_Tomato_mosaic_virus = len(os.listdir(train_dir_Tomato_mosaic_virus))
num_train_Tomato_Yellow_Leaf_Curl_Virus = len(os.listdir(train_dir_Tomato_Yellow_Leaf_Curl_Virus))

num_val_Cherry_healthy = len(os.listdir(val_dir_Cherry_healthy))
num_val_Cherry_Powdery_mildew = len(os.listdir(val_dir_Cherry_Powdery_mildew))
num_val_Grape_healthy = len(os.listdir(val_dir_Grape_healthy))
num_val_Grape_Black_rot = len(os.listdir(val_dir_Grape_Black_rot))
num_val_Grape_Esca_Black_Measles = len(os.listdir(val_dir_Grape_Esca_Black_Measles))
num_val_Grape_Leaf_blight = len(os.listdir(val_dir_Grape_Leaf_blight))
num_val_Tomato_healthy = len(os.listdir(val_dir_Tomato_healthy))
num_val_Tomato_Bacterial_Spot = len(os.listdir(val_dir_Tomato_Bacterial_Spot))
num_val_Tomato_Early_blight = len(os.listdir(val_dir_Tomato_Early_blight))
num_val_Tomato_Late_blight = len(os.listdir(val_dir_Tomato_Late_blight))
num_val_Tomato_Leaf_Mold = len(os.listdir(val_dir_Tomato_Leaf_Mold))
num_val_Tomato_Septoria_leaf_spot = len(os.listdir(val_dir_Tomato_Septoria_leaf_spot))
num_val_Tomato_Spider_mites = len(os.listdir(val_dir_Tomato_Spider_mites))
num_val_Tomato_Target_Spot = len(os.listdir(val_dir_Tomato_Target_Spot))
num_val_Tomato_mosaic_virus = len(os.listdir(val_dir_Tomato_mosaic_virus))
num_val_Tomato_Yellow_Leaf_Curl_Virus = len(os.listdir(val_dir_Tomato_Yellow_Leaf_Curl_Virus))


total_train =  num_train_Cherry_healthy + num_train_Cherry_Powdery_mildew + num_train_Grape_Black_rot + num_train_Grape_Esca_Black_Measles \
               + num_train_Grape_Leaf_blight + num_train_Grape_healthy + num_train_Tomato_Bacterial_Spot + num_train_Tomato_Early_blight + num_train_Tomato_Late_blight\
               + num_train_Tomato_Leaf_Mold + num_train_Tomato_Septoria_leaf_spot + num_train_Tomato_Spider_mites + num_train_Tomato_Target_Spot + num_train_Tomato_Yellow_Leaf_Curl_Virus\
               + num_train_Tomato_healthy + num_train_Tomato_mosaic_virus
print('Total train images: ', total_train)

total_val = num_val_Cherry_healthy + num_val_Cherry_Powdery_mildew + num_val_Grape_Black_rot + num_val_Grape_Esca_Black_Measles\
            + num_val_Grape_Leaf_blight + num_val_Grape_healthy + num_val_Tomato_Bacterial_Spot + num_val_Tomato_Early_blight + num_val_Tomato_Late_blight\
            + num_val_Tomato_Leaf_Mold + num_val_Tomato_Septoria_leaf_spot + num_val_Tomato_Spider_mites + num_val_Tomato_Target_Spot \
            + num_val_Tomato_Yellow_Leaf_Curl_Virus + num_val_Tomato_healthy + num_val_Tomato_mosaic_virus

print('Total validation images: ', total_val)

BATCH_SIZE = 200
IMG_SHAPE = 256

def plotImages(images_arr):
    # plot array of images with 1 row and 5 columns
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

image_gen_train = ImageDataGenerator(rescale=1./255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     fill_mode='nearest')


train_data_gen = image_gen_train.flow_from_directory(target_size=(128, 128),
                                                     batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     class_mode='categorical')

# print('test data gen: ', test_data_gen[0][0])
# plotImages(test_data_gen[0][0])

# example of how looks single image five times with random augmentations
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen_val = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

val_data_gen = image_gen_val.flow_from_directory(target_size=(128, 128),
                                                 batch_size=BATCH_SIZE,
                                                 directory=val_dir,
                                                 class_mode='categorical')

print(val_data_gen)

with tf.device('/gpu:0'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
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
with tf.device('/gpu:0'):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model.summary()

#Log directory for augmented model
log_dir = ".\\logs\\test\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)



# History for Augmented model
print('--------------Training augmented model----------------')
with tf.device('/gpu:0'):
    history = model.fit_generator(train_data_gen,
                                  epochs=EPOCHS,
                                  steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
                                  validation_data=val_data_gen,
                                  validation_steps=int(np.ceil(total_val / float(BATCH_SIZE))),
                                  callbacks=[tensorboard_callback])


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
plt.title('Training and Validation Accuracy Augmented Model')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss Augmented Model')
plt.show()

#----------------------BATCH NORMALIZED MODEL------------------------
# load model from BatchNormModel.py
# batchNormModel = BNM.batchNormModel()
# batchNormModel.compile(optimizer='adam',
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# batchNormModel.summary()

# log directory for BNM
# log_dir = ".\\logs\\test\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 'BNM'
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# History for Batch Normalized model
# print('--------------Training batch normalized model----------------')
# BNHistory = model.fit_generator(train_data_gen,
#                               epochs=EPOCHS,
#                               steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
#                               validation_data=val_data_gen,
#                               validation_steps=int(np.ceil(total_val / float(BATCH_SIZE))),
#                               verbose=2,
#                               callbacks=[tensorboard_callback])
#
#
# acc = BNHistory.history['accuracy']
# val_acc = BNHistory.history['val_accuracy']
# loss = BNHistory.history['loss']
# val_loss = BNHistory.history['val_loss']
#
# epochs_range = range(EPOCHS)
#
# plt.figure(figsize=(8,8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy BNModel')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss BNModel')
# plt.show()

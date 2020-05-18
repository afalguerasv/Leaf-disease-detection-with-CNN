from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import datetime
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sn

import graphPlot as graphplt
import BatchNormModel as BNM
import L2RegModel as l2
import DropoutModel as dptmodel

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

print(tf.version.VERSION, tf.executing_eagerly(), tf.keras.layers.BatchNormalization._USE_V2_BEHAVIOR)

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

base_dir = os.path.dirname('B:\TFG\PlantVillage2\\')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
targetNames = ['Cherry___Powdery_mildew', 'Cherry___healthy', 'Grape___Black_rot',
               'Grape___Esca_Black_Measles', 'Grape___Leaf_blight_Isariopsis_Leaf_Spot', 'Grape___healthy',
               'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
               'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
shortNames = ['Ch_PM', 'Ch_healthy', 'Gr_BlRt', 'Gr_EsBlMe', 'Gr_LeafBl', 'Gr_healthy',
               'To_BaSpt', 'To_EarBl', 'To_LaBl', 'To_Leaf_Mo', 'To_SeptLS',
               'To_Spider', 'To_TarSpt',
                'To_YeLeafCuVi', 'To_MVirus', 'To_healthy']

EPOCHS = 25
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

image_gen_train = ImageDataGenerator(rescale=1./255,
                                     rotation_range=40,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     fill_mode='nearest')


train_data_gen = image_gen_train.flow_from_directory(target_size=(100, 100),
                                                     batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     color_mode='rgb',
                                                     shuffle=True,
                                                     class_mode='categorical')

# # # example of how looks single image five times with random augmentations
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(target_size=(100, 100),
                                                 batch_size=BATCH_SIZE,
                                                 directory=val_dir,
                                                 color_mode='rgb',
                                                 shuffle=False,
                                                 class_mode='categorical')

image_gen_test = ImageDataGenerator(rescale=1./255)

test_data_gen = image_gen_val.flow_from_directory(target_size=(100, 100),
                                                 batch_size=BATCH_SIZE,
                                                 directory=test_dir,
                                                 shuffle=False,
                                                 class_mode='categorical')
print('classes: ', test_data_gen.classes)
print('index de les classes: ', test_data_gen.class_indices)
# with tf.device('/gpu:0'):
#     augmentedModel = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
#         tf.keras.layers.MaxPooling2D(2, 2),
#
#         tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),
#
#         tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),
#
#         tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),
#
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.Dense(16)
#     ])
#
# with tf.device('/gpu:0'):
#     augmentedModel.compile(optimizer='adam',
#                            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#                            metrics=['accuracy'])
# #
# augmentedModel.summary()
#
# #Log directory for augmented model
# log_dir = ".\\logs\\test\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "AugmentedOnlyTrain"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#
# # History for Augmented model
# print('--------------Training augmented model----------------')
# with tf.device('/gpu:0'):
#     history = augmentedModel.fit_generator(train_data_gen,
#                                   epochs=EPOCHS,
#                                   steps_per_epoch=int(np.ceil(total_train_images / float(BATCH_SIZE))),
#                                   validation_data=val_data_gen,
#                                   validation_steps=int(np.ceil(total_validation_images / float(BATCH_SIZE))),
#                                   callbacks=[tensorboard_callback])
#
# augmentedModel.save('.\\models\\SimpleModel\\augmentedModel.h5')
# acc = history.history['accuracy']
# print('acc ', acc)
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(EPOCHS)
# print('epochs range: ', epochs_range)
#
# # plot accuracy during training
# plt.subplot(2, 1, 2)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.grid()
# plt.xticks(range(0, EPOCHS-1))
# # plot loss during training
# plt.subplot(2, 1, 1)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xticks(range(0, EPOCHS-1))
# plt.grid()
# plt.show()
#
# #Confution Matrix and Classification Report
# Y_pred = augmentedModel.predict_generator(test_data_gen, total_validation_images // BATCH_SIZE+1)
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# print(confusion_matrix(test_data_gen.classes, y_pred))
# print('Classification Report')
#
# print(classification_report(test_data_gen.classes, y_pred, target_names=targetNames))
#
# df_cm = pd.DataFrame(confusion_matrix(test_data_gen.classes, y_pred), index=[i for i in shortNames],
#                      columns= [i for i in shortNames])
# plt.figure(figsize=(10, 7))
# sn.heatmap(df_cm, annot=True, cmap='coolwarm', fmt="d")
# plt.title('Augmented Model')
# plt.show()

#----------------------BATCH NORMALIZED MODEL------------------------
# load model from BatchNormModel.py
batchNormModel = BNM.batchNormModel()
batchNormModel.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
batchNormModel.summary()

# log directory for BNM
log_dir = ".\\logs\\test\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 'BNM'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# History for Batch Normalized model
print('--------------Training batch normalized model----------------')
with tf.device('/gpu:0'):
    BNHistory = batchNormModel.fit_generator(train_data_gen,
                                             epochs=EPOCHS,
                                             steps_per_epoch=int(np.ceil(total_train_images / float(BATCH_SIZE))),
                                             validation_data=val_data_gen,
                                             validation_steps=int(np.ceil(total_validation_images / float(BATCH_SIZE))),
                                             verbose=2,
                                             callbacks=[tensorboard_callback])

batchNormModel.save('.\\models\\SimpleModel\\batchNormModel.h5')
acc = BNHistory.history['accuracy']
val_acc = BNHistory.history['val_accuracy']
loss = BNHistory.history['loss']
val_loss = BNHistory.history['val_loss']

epochs_range = range(EPOCHS)

graphplt.plotGraph(acc, val_acc, loss, val_loss, epochs_range,
                   'Training and Validation Accuracy BNModel',
                   'Training and Validation Loss BNModel',
                   0,
                   EPOCHS-1)

#Confution Matrix and Classification Report
Y_pred = batchNormModel.predict_generator(test_data_gen, total_validation_images // BATCH_SIZE+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_data_gen.classes, y_pred))
print('Classification Report')

print(classification_report(test_data_gen.classes, y_pred, target_names=targetNames))

df_cm = pd.DataFrame(confusion_matrix(test_data_gen.classes, y_pred), index=[i for i in shortNames],
                     columns= [i for i in shortNames])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, cmap='coolwarm', fmt="d")
plt.title('BatchNormalization Model')
plt.show()

# #----------------------L2 REGULARIZED MODEL------------------------
# # load model from L2RegModel.py
# l2Model = l2.L2Model()
# with tf.device('/gpu:0'):
#     l2Model.compile(optimizer='adam',
#                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#                    metrics=['accuracy'])
#
# l2Model.summary()
#
# # log directory for L2 model
# log_dir = ".\\logs\\test\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 'L2_0_001_noBatch'
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#
# # History for Batch L2 model
# print('--------------Training L2 model----------------')
# L2History = l2Model.fit_generator(train_data_gen,
#                               epochs=EPOCHS,
#                               steps_per_epoch=int(np.ceil(total_train_images / float(BATCH_SIZE))),
#                               validation_data=val_data_gen,
#                               validation_steps=int(np.ceil(total_validation_images / float(BATCH_SIZE))),
#                               verbose=2,
#                               callbacks=[tensorboard_callback])
#
# l2Model.save('.\\models\\SimpleModel\\L2Model.h5')
#
# acc = L2History.history['accuracy']
# val_acc = L2History.history['val_accuracy']
# loss = L2History.history['loss']
# val_loss = L2History.history['val_loss']
#
# epochs_range = range(EPOCHS)
#
# graphplt.plotGraph(acc, val_acc, loss, val_loss, epochs_range,
#                    'Training and Validation Accuracy L2Model',
#                    'Training and Validation Loss L2Model',
#                     0,
#                     EPOCHS-1)
# #Confution Matrix and Classification Report
# Y_pred = l2Model.predict_generator(test_data_gen, total_validation_images // BATCH_SIZE+1)
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# print(confusion_matrix(test_data_gen.classes, y_pred))
# print('Classification Report')
#
# print(classification_report(test_data_gen.classes, y_pred, target_names=targetNames))
#
# df_cm = pd.DataFrame(confusion_matrix(test_data_gen.classes, y_pred), index=[i for i in shortNames],
#                      columns= [i for i in shortNames])
# plt.figure(figsize=(10, 7))
# sn.heatmap(df_cm, annot=True, cmap='coolwarm', fmt="d")
# plt.title('L2 Model')
# plt.show()

# #----------------------DROPOUT MODEL------------------------
#
# # load model from DropoutModel
# dptmodel = dptmodel.dropoutModel()
# dptmodel.compile(optimizer='adam',
#                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#                metrics=['accuracy'])
# dptmodel.summary()
#
# # log directory for L2 model
# log_dir = ".\\logs\\test\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 'DPT_0_05_noBatch'
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#
# # History for Dropout model
# print('--------------Training Dropout model----------------')
# DPTHystory = dptmodel.fit_generator(train_data_gen,
#                               epochs=EPOCHS,
#                               steps_per_epoch=int(np.ceil(total_train_images / float(BATCH_SIZE))),
#                               validation_data=val_data_gen,
#                               validation_steps=int(np.ceil(total_validation_images / float(BATCH_SIZE))),
#                               verbose=2,
#                               callbacks=[tensorboard_callback])
#
# dptmodel.save('.\\models\\SimpleModel\\DptModel0_4_withBatchAxis.h5')
#
# acc = DPTHystory.history['accuracy']
# val_acc = DPTHystory.history['val_accuracy']
# loss = DPTHystory.history['loss']
# val_loss = DPTHystory.history['val_loss']
#
# epochs_range = range(EPOCHS)
#
# graphplt.plotGraph(acc, val_acc, loss, val_loss, epochs_range,
#                    'Training and Validation Accuracy DPTModel  04 axis',
#                    'Training and Validation Loss DPTModel',
#                    0,
#                    EPOCHS-1)
#
# Y_pred = dptmodel.predict_generator(test_data_gen, total_validation_images // BATCH_SIZE+1)
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# print(confusion_matrix(test_data_gen.classes, y_pred))
# print('Classification Report')
#
# print(classification_report(test_data_gen.classes, y_pred, target_names=targetNames))
#
# df_cm = pd.DataFrame(confusion_matrix(test_data_gen.classes, y_pred), index=[i for i in shortNames],
#                      columns= [i for i in shortNames])
# plt.figure(figsize=(10, 7))
# sn.heatmap(df_cm, annot=True, cmap='coolwarm', fmt="d")
# plt.title('Dropout Model')
# plt.show()

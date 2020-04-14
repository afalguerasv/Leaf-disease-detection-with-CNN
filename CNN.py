import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

base_dir = os.path.dirname('B:\TFG\PlantVillage\\')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

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

print('Total train images: ',  num_train_Cherry_healthy + num_train_Cherry_Powdery_mildew + num_train_Grape_Black_rot + num_train_Grape_Esca_Black_Measles
      + num_train_Grape_Leaf_blight + num_train_Grape_healthy + num_train_Tomato_Bacterial_Spot + num_train_Tomato_Early_blight + num_train_Tomato_Late_blight
      + num_train_Tomato_Leaf_Mold + num_train_Tomato_Septoria_leaf_spot + num_train_Tomato_Spider_mites + num_train_Tomato_Target_Spot + num_train_Tomato_Yellow_Leaf_Curl_Virus
      + num_train_Tomato_healthy + num_train_Tomato_mosaic_virus)

print('Total validation images: ', num_val_Cherry_healthy + num_val_Cherry_Powdery_mildew + num_val_Grape_Black_rot + num_val_Grape_Esca_Black_Measles
      + num_val_Grape_Leaf_blight + num_val_Grape_healthy + num_val_Tomato_Bacterial_Spot + num_val_Tomato_Early_blight + num_val_Tomato_Late_blight
      + num_val_Tomato_Leaf_Mold + num_val_Tomato_Septoria_leaf_spot + num_val_Tomato_Spider_mites + num_val_Tomato_Target_Spot + num_val_Tomato_Yellow_Leaf_Curl_Virus
      + num_val_Tomato_healthy + num_val_Tomato_mosaic_virus)

BATCH_SIZE = 100
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
                                     rotation_range=45,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE),
                                                     class_mode='binary')

# example of how looks single image five times with random augmentations
# augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=val_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='binary')


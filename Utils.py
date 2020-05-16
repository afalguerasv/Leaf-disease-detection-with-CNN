
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def simpleImageTrain():
    ImageDataGenerator(rescale=1. / 255)

def complexImageTrain():
    ImageDataGenerator(rescale=1. / 255,
                       rotation_range=40,
                       width_shift_range=0.2,
                       height_shift_range=0.2,
                       shear_range=0.2,
                       zoom_range=0.2,
                       horizontal_flip=True,
                       vertical_flip=True,
                       fill_mode='nearest')
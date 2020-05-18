import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn

BATCH_SIZE = 32

base_dir = os.path.dirname('B:\TFG\PlantVillage2\\')
test_dir = os.path.join(base_dir, 'Test')
val_dir = os.path.join(base_dir, 'validation')
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

total_validation_images = 0
total_test_images = 1440
for dirpath, dirnames, filenames in os.walk(val_dir):
    N_c = len(filenames)
    total_validation_images += N_c
    print("Files in ", dirpath, N_c)
print("Total Validation Files ",total_validation_images)

image_gen_val = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

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

simpleModel = tf.keras.models.load_model('.\\models\\SimpleModel\\simpleModel.h5')

augmentedModel = tf.keras.models.load_model('.\\models\\SimpleModel\\augmentedModel.h5')

batchNormModel = tf.keras.models.load_model('.\\models\\SimpleModel\\batchNormModel.h5')

L2Model = tf.keras.models.load_model('.\\models\\SimpleModel\\L2Model.h5')


# # -------------------SIMPLE MODEL---------------------------
# # PREDICT THE OUTPUT FOR TEST FOLDER
STEP_SIZE_TEST = test_data_gen.n//test_data_gen.batch_size
# test_data_gen.reset()
# pred = simpleModel.predict_generator(test_data_gen, steps=STEP_SIZE_TEST, verbose=1)
# predicted_class_indices=np.argmax(pred, axis=1)
# labels = (val_data_gen.class_indices)
# labels = dict((v, k) for k, v in labels.items())
# predictions = [labels[k] for k in predicted_class_indices]
#
# # SAVING TO CSV
# filenames = test_data_gen.filenames
# results = pd.DataFrame({"Filename":filenames,
#                         "Predictions":predictions})
# results.to_csv("resultsSimpleModel2.csv", index=False)

#Confution Matrix and Classification Report
Y_pred = batchNormModel.predict_generator(test_data_gen, steps=STEP_SIZE_TEST, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# print(confusion_matrix(test_data_gen.classes, y_pred))
print('Classification Report')

print(classification_report(test_data_gen.classes, y_pred, target_names=targetNames))

df_cm = pd.DataFrame(confusion_matrix(test_data_gen.classes, y_pred), index=[i for i in shortNames],
                     columns= [i for i in shortNames])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, cmap='coolwarm', fmt="d")
plt.show()

# --------------------AUGMENTED MODEL----------------------------

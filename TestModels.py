import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

BATCH_SIZE = 32

base_dir = os.path.dirname('B:\TFG\PlantVillage2\\')
test_dir = os.path.join(base_dir, 'Test')
val_dir = os.path.join(base_dir, 'validation')

image_gen_train = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_train.flow_from_directory(target_size=(128, 128),
                                                     batch_size=BATCH_SIZE,
                                                     directory=val_dir,
                                                     shuffle=True,
                                                     class_mode='categorical')
test_data_gen = image_gen_train.flow_from_directory(target_size=(128, 128),
                                                     batch_size=1,
                                                     directory=test_dir,
                                                     shuffle=False,
                                                     class_mode=None)

simpleModel = tf.keras.models.load_model('.\\models\\SimpleModel\\simpleModel.h5')
simpleModel.summary()

# -------------------SIMPLE MODEL---------------------------
# PREDICT THE OUTPUT FOR TEST FOLDER
STEP_SIZE_TEST = test_data_gen.n//test_data_gen.batch_size
test_data_gen.reset()
pred = simpleModel.predict_generator(test_data_gen, steps=STEP_SIZE_TEST, verbose=1)
predicted_class_indices=np.argmax(pred, axis=1)
labels = (val_data_gen.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# SAVING TO CSV
filenames = test_data_gen.filenames
results = pd.DataFrame({"Filename":filenames,
                        "Predictions":predictions})
results.to_csv("resultsSimpleModel2.csv", index=False)

# --------------------AUGMENTED MODEL----------------------------

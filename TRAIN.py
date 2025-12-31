import  os
import cv2

import shutil
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16,preprocess_input
from sklearn.model_selection import train_test_split

file_path =r"C:\Users\sivan\Desktop\ML_DATASET\DL_DATASET\CNN_REGRESSION\FACE_AGE"
ages = []
images = []

image_path = os.listdir(file_path)
for image in image_path:
    age = int(image.split("_")[0])
    ages.append(age)
    img_full_path = os.path.join(file_path, image)
    img = cv2.imread(img_full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2 .resize(img,(224,224))
    img = preprocess_input(img)
    images.append(img)


# for image ,age  in zip(images , ages):
#     data.append([age ,image] )



vgg = VGG16(include_top=False,input_shape=(224,224,3),weights='imagenet')

# stop training exsisiting weight
# ===================================================
for layer in vgg.layers[:-4]:
    layer.trainable = False
for layer in vgg.layers[-4:]:
    layer.trainable = True


# build custom layers:
# =========================================
x = vgg.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='linear')(x)

# MODEL_BUILD:
# ============================================
model =Model(inputs=vgg.input,outputs=output)
model.compile (optimizer=Adam(learning_rate=1e-5),loss="mse",metrics=["mae"])

# TRAIN_TEST_SPLIT
# =============================================
x = np.array(images, dtype="float32")
y = np.array(ages, dtype="float32") / 100.0


x_train ,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=42)
model_trianed = model.fit(x_train,y_train,epochs=5 , validation_data=(x_test,y_test))

# SAVE THE MODEL:
# ====================================
model.save(r"C:\Users\sivan\Desktop\ML_DATASET\DL_DATASET\CNN_REGRESSION\AGE_PRED_BY_IMAGE.h5")

print("MODEL_HAS_SAVED")
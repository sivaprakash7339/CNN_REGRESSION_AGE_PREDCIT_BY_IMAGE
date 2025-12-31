import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input


# LOAD_MODEL:
# ========================================================
save_path = r"C:\Users\sivan\Desktop\ML_DATASET\DL_DATASET\CNN_REGRESSION\AGE_PRED_BY_IMAGE.h5"
model = load_model(save_path)

# INPUT:
# ========================================================
user_input = cv2.imread(r"C:\Users\sivan\Desktop\ML_DATASET\DL_DATASET\CNN_REGRESSION\CNN_TEST_IMAGE.jpg")

# PREPROCESS IMAGE (SAME AS TRAINING)
user_input = cv2.cvtColor(user_input, cv2.COLOR_BGR2RGB)
user_input = cv2.resize(user_input, (224, 224))
user_input = preprocess_input(user_input)

# IMAGE_TO_ARRAY:
# ==========================================================
user_input = np.expand_dims(user_input, axis=0)

# PREDICTION:
out = model.predict(user_input)

print("PREDICTED AGE:", round(out[0][0]*100, 2))



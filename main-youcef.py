import matplotlib
import sklearn
from Reconstruction import Reconstruction
from CONFIGURATION import CONFIGURATION
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# -----------------------------
# ignorable
print("TensorFlow version:", tf.__version__)
print("cv2 version:", cv2.__version__)
print("np version:", np.__version__)
# ignorable
print("matplotlib version:", matplotlib.__version__)
print("sklearn version:", sklearn.__version__)

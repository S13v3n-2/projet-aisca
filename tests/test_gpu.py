import torch
print(torch.cuda.is_available())

import tensorflow as tf
import os
import subprocess

print("===== TensorFlow =====")
print("TF version:", tf.__version__)
print("TF built with CUDA:", tf.sysconfig.get_build_info()['cuda_version'])
print("TF built with cuDNN:", tf.sysconfig.get_build_info()['cudnn_version'])

print("\n===== GPU Devices =====")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU detected by TensorFlow.")

print("\n===== CUDA version installed on system =====")
try:
    nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
    print(nvcc_version)
except Exception as e:
    print("nvcc not found:", e)

print("\n===== cuDNN version installed on system =====")
# Vérifie si le fichier cudnn_version.h existe
cudnn_header = "/usr/include/cudnn_version.h"
if os.path.exists(cudnn_header):
    with open(cudnn_header, "r") as f:
        for line in f:
            if "CUDNN" in line:
                print(line.strip())
else:
    print("cuDNN header not found at /usr/include/cudnn_version.h")


"""import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))"""

import tensorflow as tf
with tf.device('/GPU:0'):
    a = tf.random.normal([4000, 4000])
    b = tf.matmul(a, a)
print("OK GPU")

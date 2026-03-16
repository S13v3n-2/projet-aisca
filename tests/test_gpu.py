# petit script pour vérifier que le GPU est bien détecté
# on l'a utilisé au début du projet pour s'assurer que CUDA marchait
# pas sûr que ce soit encore utile mais on le garde au cas où

import torch
print(torch.cuda.is_available())

import tensorflow as tf
import os
import subprocess

print("TensorFlow")
print("TF version:", tf.__version__)
print("TF built with CUDA:", tf.sysconfig.get_build_info()['cuda_version'])
print("TF built with cuDNN:", tf.sysconfig.get_build_info()['cudnn_version'])

print("\nGPU Devices")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU detected by TensorFlow.")

print("\nCUDA version")
try:
    nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
    print(nvcc_version)
except Exception as e:
    print("nvcc not found:", e)

# on check aussi cuDNN, des fois c'est installé mais pas linké correctement
print("\ncuDNN version")
cudnn_header = "/usr/include/cudnn_version.h"
if os.path.exists(cudnn_header):
    with open(cudnn_header, "r") as f:
        for line in f:
            if "CUDNN" in line:
                print(line.strip())
else:
    print("cuDNN header not found at /usr/include/cudnn_version.h")


# test rapide de calcul GPU pour vérifier que ça tourne vraiment
import tensorflow as tf
with tf.device('/GPU:0'):
    a = tf.random.normal([4000, 4000])
    b = tf.matmul(a, a)
print("OK GPU")

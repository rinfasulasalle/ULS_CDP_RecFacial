import cv2
import numpy as np
cv2.__version__
# Cargar imagen en CPU
image = cv2.imread('imagen.jpg')

# Crear objeto GpuMat a partir de la imagen
gpu_image = cv2.cuda_GpuMat()
gpu_image.upload(image)

# Convertir a escala de grises en la GPU
gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)

# Descargar la imagen resultante de la GPU a la CPU
gray = gpu_gray.download()

# Mostrar la imagen en escala de grises
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

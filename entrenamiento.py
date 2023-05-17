import cv2
import os
import numpy as np

dataPath = 'data'
peopleList = os.listdir(dataPath)
print("Personas: ", peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo imgs')
    
    for fileName in os.listdir(personPath):
        # print('Rostros: ',nameDir + '/' + fileName)
        labels.append(label)
    
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))
        image = cv2.imread(personPath + '/' + fileName, 0)

        # ---------------- enseño todas las imgs que tengo
        # cv2.imshow('image',image)
        # cv2.waitKey(10)
    label = label + 1
#----------------
cv2.destroyAllWindows()

#-------------- Cantidad fotos
# print('labels', labels)
# print('num etiquetas 0: ', np.count_nonzero(np.array(labels) == 0))
# print('num etiquetas 0: ', np.count_nonzero(np.array(labels) == 1))
# print(len(facesData))

# Utilizaremos aca el   EigenFaceRecognizer
#                       FisherFaceRecognizer
#                       LBPHFaceRecognizer  
face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer()
# face_recognizer = cv2.face.LBPHFaceRecognizer()
# Entrenamos el algoritmo
print('Entrenando...')
face_recognizer.train(facesData, np.array(labels))

# Guardamos el modelo
face_recognizer.write('modeloFaceFrontalData2023.xml')
print('Modelo Guardado')



import cv2
import os

dataPath = 'data'
imagePaths = os.listdir(dataPath)
print('imagPath=',imagePaths)


face_recognizer = cv2.face.EigenFaceRecognizer_create()


face_recognizer.read('modeloFaceFrontalData2023.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# definimos model0
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # camabiamos a escala de grises
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    #programamos cuadra que se vera en cara y el texcto
    for (x, y, w, h) in faces:
        rostro = auxFrame[y: y+h, x: x+w]
        rostro = cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
        #rostro = cv2.resize(rostro, 150, 150, interpolation= cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255,255,0), 1, cv2.LINE_AA)

        # si persona en camara esta en base de datos
        if result[1] < 580:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y -25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        else:
            cv2.putText(frame, 'Persona Desconocida', (x,y-20), 2, 0.8, (0,0,255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
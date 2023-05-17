import cv2
import os
import imutils


# Pra crear usuarios
personName = 'roger'
dataPath = 'data'
personPath = dataPath + '/' + personName
# ---------------------------------------------


# Condicional par crear carpeta
if not os.path.exists(personPath):
    os.makedirs(personPath)
    print('Usuario Creado: ', personPath)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# -------------
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0


while True:
    ret, frame =  cap.read()

    if ret == False:
        break
    frame = imutils.resize(frame, width = 320) # Redimensiona imagen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # corrected code
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,00), 2)
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (720, 720), interpolation= cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
        #cv2.imwrite(personPath + '/' + str(count) + '.jpg', rostro)
        count = count + 1

    cv2.imshow('datpy', frame)

    if cv2.waitKey(1) == 27 or  count >=200 : 
        break
cap.release()
cv2.destroyAllWindows()

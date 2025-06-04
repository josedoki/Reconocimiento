import cv2
import os
import imutils

# Nombre de la persona a capturar
personName = "Juarez"
dataPath = 'C:/Users/josan/OneDrive/Escritorio/TRABAJOS EMI/8vo semestre ICI/copiladores/rostros/Data'
personPath = os.path.join(dataPath, personName)

# Crear carpeta si no existe
if not os.path.exists(personPath):
    print('Carpeta creada:', personPath)
    os.makedirs(personPath)

# Abrir video
cap = cv2.VideoCapture('juarez2.mp4')

# Asegurarse de usar el nombre correcto del clasificador
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 701

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Guardar rostro recortado
        rostro = frame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150))
        cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)

        # Dibujar rectángulo
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        count += 1

    # Mostrar el video en tiempo real
    cv2.imshow('Captura de Rostros', frame)

    # Salir con ESC o si se capturaron 200 rostros
    k = cv2.waitKey(1)
    if k == 27 or count >= 800:
        break

cap.release()
cv2.destroyAllWindows()
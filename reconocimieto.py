import cv2
import os


dataPath = 'C:/Users/josan/OneDrive/Escritorio/TRABAJOS EMI/8vo semestre ICI/copiladores/rostros'


names_list = ["Canedo", "Juarez"]
print('Lista de personas para el mapeo:', names_list)

# --- INICIALIZACIÓN DEL RECONOCEDOR Y CLASIFICADOR ---
face_recognizer = cv2.face.EigenFaceRecognizer.create()

# Leyendo el modelo entrenado
try:
    face_recognizer.read('modeloEigenFace1.xml')
    print("Modelo 'modeloEigenFace.xml' cargado exitosamente.")
except cv2.error as e:
    print(f"Error al cargar el modelo: {e}")
    print("Asegúrate de que 'modeloEigenFace.xml' esté en el mismo directorio que tu script.")
    exit() 

# Carga el video a procesar
cap = cv2.VideoCapture('videos/juarez.mp4')

# Carga el clasificador en cascada para detección facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read() # Lee un fotograma del video
    if not ret: # Si no se pudo leer el fotograma (fin del video o error), salimos
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convierte el fotograma a escala de grises
    auxFrame = gray.copy() # Copia del fotograma en gris para extraer rostros

    # Detecta rostros en el fotograma en escala de grises
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extrae la región del rostro
        rostro = auxFrame[y:y + h, x:x + w]
        # Redimensiona el rostro a 150x150 píxeles, tamaño esperado por el modelo
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        # Para EigenFace, una confianza más BAJA significa una mejor coincidencia.
        result = face_recognizer.predict(rostro)

        # Si la confianza es baja (buena coincidencia), mostramos el nombre.
        if result[1] < 5700: 
            # Accede al nombre de la persona usando el ID predicho (result[0])
            person_name = names_list[result[0]]
            
            # Muestra el nombre reconocido en verde
            cv2.putText(frame, person_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            # Dibuja un rectángulo verde alrededor del rostro reconocido
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # Si la confianza es alta (mala coincidencia), mostramos 'DESCONOCIDO' en rojo
            cv2.putText(frame, 'DESCONOCIDO', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            # Dibuja un rectángulo rojo alrededor del rostro desconocido
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Reconocimiento Facial', frame) # Muestra el fotograma procesado

    # Espera 1 milisegundo por una pulsación de tecla
    k = cv2.waitKey(1)
    if k == 27: # Si la tecla presionada es 'Esc' (ASCII 27), salimos del bucle
        break

# --- LIBERACIÓN DE RECURSOS ---
cap.release() # Libera el objeto de captura de video
cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV
import cv2
import os
import numpy as np

dataPath = 'C:/Users/josan/OneDrive/Escritorio/TRABAJOS EMI/8vo semestre ICI/copiladores/rostros/Data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
faceData = []

label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo imagendes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir)
        labels.append(label)
        faceData.append(cv2.imread(personPath + '/' + fileName, 0))
        image = cv2.imread(personPath+'/'+fileName,0)

    label = label + 1

face_recognizer = cv2.face.EigenFaceRecognizer.create()

#Entrenando el reconocedor de rostros
print("Enterenando...")
face_recognizer.train(faceData, np.array(labels))

#Almacenando el modelo obtenido
face_recognizer.write('modeloEigenFace1.xml')
print("Modelo almacenado...")
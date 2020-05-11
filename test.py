import os
import shutil
import random
from pathlib import Path

dirpath = 'B:/TFG/plantvillage-dataset/plantvillage dataset/color'
destTrainPath = 'B:\TFG\PlantVillage\Train\\'
destValiPath = 'B:\TFG\PlantVillage\Validation\\'
destTestPath = 'B:\TFG\Test\\'
count = 0
arrayLabel = []
for x in os.walk('B:\TFG\DatasetTest'):
    # X[0] fa una primera iteració amb B:\TFG\Dataset2 i la resta retorna el nom de cada subcarpeta
    print(x[0])
    #x[2] retorna els elements de la subcarpeta
    print(len(x[2]))
    # Calculem el numero de imatges de cada tipus en % per cada cas
    numTrain = int((len(x[2])*70)/100)
    numValidation = int((len(x[2])*10)/100)
    numTest = int((len(x[2])*20)/100)
    print('numTrain: ', numTrain)
    print('numValidatio: ', numValidation)
    print('numTest: ', numTest)
    # Total ens serveix per validar que no amb l'arrodoniment del % no agafem més imatges de les totals
    print('Total: ', numTest+numTrain+numValidation)

    # utilitzem os.listdir(<path>) per buscar la carpeta d'origen i seleccionar numTrain elements

    testFilenames = random.sample(os.listdir(Path(x[0])), numTest)

    for testImage in testFilenames:
        shutil.copy(x[0] + '\\' + testImage, destTestPath + 'Test')
        tail = os.path.split(x[0])
        # en aquest punt tail[1] conte el nom de la carpeta que al seu temps es el label de la imatge
        if tail[1] == 'Cherry___healthy':
            arrayLabel.append(0)
        elif tail[1] == 'Cherry___Powdery_mildew':
            arrayLabel.append(1)
        elif tail[1] == 'Grape___healthy':
            arrayLabel.append(2)
        elif tail[1] == 'Grape___Black_rot':
            arrayLabel.append(3)
        elif tail[1] == 'Grape___Esca_Black_Measles':
            arrayLabel.append(4)
        elif tail[1] == 'Grape___Leaf_blight_Isariopsis_Leaf_Spot':
            arrayLabel.append(5)
        elif tail[1] == 'Tomato___healthy':
            arrayLabel.append(6)
        elif tail[1] == 'Tomato___Bacterial_spot':
            arrayLabel.append(7)
        elif tail[1] == 'Tomato___Early_blight':
            arrayLabel.append(8)
        elif tail[1] == 'Tomato___Late_blight':
            arrayLabel.append(9)
        elif tail[1] == 'Tomato___Leaf_Mold':
            arrayLabel.append(10)
        elif tail[1] == 'Tomato___Septoria_leaf_spot':
            arrayLabel.append(11)
        elif tail[1] == 'Tomato___Spider_mites_Two-spotted_spider_mite':
            arrayLabel.append(12)
        elif tail[1] == 'Tomato___Target_Spot':
            arrayLabel.append(13)
        elif tail[1] == 'Tomato___Tomato_mosaic_virus':
            arrayLabel.append(14)
        elif tail[1] == 'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
            arrayLabel.append(15)

    count += 1

print(arrayLabel)


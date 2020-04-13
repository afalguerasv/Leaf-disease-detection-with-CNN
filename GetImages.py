import os
import shutil
import random
from pathlib import Path

dirpath = 'B:/TFG/plantvillage-dataset/plantvillage dataset/color'
destTrainPath = 'B:\TFG\PlantVillage\Train\\'
destValiPath = 'B:\TFG\PlantVillage\Validation\\'
destTestPath = 'B:\TFG\PlantVillage\Test\\'
count = 0
for x in os.walk('B:\TFG\Dataset2'):
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
    trainFilenames = random.sample(os.listdir(Path(x[0])), numTrain)
    for trainImg in trainFilenames:
        # creem el subdirectori en el nou path si no existeix prèviament
        if not os.path.isdir(destTrainPath + next(os.walk('B:\TFG\Dataset2'))[1][count-1]):
            os.makedirs(destTrainPath + next(os.walk('B:\TFG\Dataset2'))[1][count-1])
        # movem els elements cap el nou directori, no els copiem i enganxem ja que sino es produirien duplicats
        # a les carpetes de validacio i test.
        # Utilitzem [count-1] perquè la primera iteracio que fa el
        # for x in os.walk('B:\TFG\Dataset2') passa per el diirectori 'B:\TFG\Dataset2' i no per una de les seves
        # subcarpetes i per tant es una iteracio que no hem de tenir en compte pel count.
        shutil.move(x[0] + '\\' + trainImg, destTrainPath + next(os.walk('B:\TFG\Dataset2'))[1][count-1])

    validationFilenames = random.sample(os.listdir(Path(x[0])), numValidation)
    for valImage in validationFilenames:
        if not os.path.isdir(destValiPath + next(os.walk('B:\TFG\Dataset2'))[1][count-1]):
            os.makedirs(destValiPath + next(os.walk('B:\TFG\Dataset2'))[1][count-1])
        shutil.move(x[0] + '\\' + valImage, destValiPath + next(os.walk('B:\TFG\Dataset2'))[1][count - 1])

    testFilenames = random.sample(os.listdir(Path(x[0])), numTest)
    for testImage in testFilenames:
        if not os.path.isdir(destTestPath + next(os.walk('B:\TFG\Dataset2'))[1][count-1]):
            os.makedirs(destTestPath + next(os.walk('B:\TFG\Dataset2'))[1][count-1])
        shutil.move(x[0] + '\\' + testImage, destTestPath + next(os.walk('B:\TFG\Dataset2'))[1][count - 1])

    count += 1


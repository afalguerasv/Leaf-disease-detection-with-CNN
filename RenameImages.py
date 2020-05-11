import os

for dirname in os.listdir('B:\TFG\PlantVillage2\Test\\'):
    count = 0
    for i, filename in enumerate(os.listdir('B:\TFG\PlantVillage2\Test\\' + dirname)):
        os.rename('B:\TFG\PlantVillage2\Test\\' + dirname + "/" + filename, 'B:\TFG\PlantVillage2\Test\\' + dirname + "/" + dirname + '_' + str(count) + ".jpg")
        count += 1

for dirname in os.listdir('B:\TFG\PlantVillage\Train\\'):
    count = 0
    for i, filename in enumerate(os.listdir('B:\TFG\PlantVillage2\Train\\' + dirname)):
        os.rename('B:\TFG\PlantVillage2\Train\\' + dirname + "/" + filename, 'B:\TFG\PlantVillage2\Train\\' + dirname + "/" + dirname + '_' + str(count) + ".jpg")
        count += 1

for dirname in os.listdir('B:\TFG\PlantVillage2\Validation\\'):
    count = 0
    for i, filename in enumerate(os.listdir('B:\TFG\PlantVillage2\Validation\\' + dirname)):
        os.rename('B:\TFG\PlantVillage2\Validation\\' + dirname + "/" + filename, 'B:\TFG\PlantVillage2\Validation\\' + dirname + "/" + dirname + '_' + str(count) + ".jpg")
        count += 1

print("END")

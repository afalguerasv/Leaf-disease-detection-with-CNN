import os

for dirname in os.listdir('B:\TFG\Dataset2\\'):
    count = 0
    for i, filename in enumerate(os.listdir('B:\TFG\Dataset2\\' + dirname)):
        os.rename('B:\TFG\Dataset2\\' + dirname + "/" + filename, 'B:\TFG\Dataset2\\'
                  + dirname + "/" + dirname + '_' + str(count) + ".jpg")
        count += 1

print("END")

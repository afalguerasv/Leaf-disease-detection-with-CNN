import os

for dirname in os.listdir('B:\TFG\Last\\'):
    count = 0
    for i, filename in enumerate(os.listdir('B:\TFG\Last\\' + dirname)):
        os.rename('B:\TFG\Last\\' + dirname + "/" + filename, 'B:\TFG\Last\\'
                  + dirname + "/" + dirname + '_' + str(count) + ".jpg")
        count += 1

print("END")

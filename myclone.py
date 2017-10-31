import csv
import cv2
import numpy as np
MYCSV="record/driving_log.csv"
IMAGEDIR="record/IMG/"
FILEDELIM ="\\"
USENVIDIA = True
EPOCHS=7
lines = []

with open(MYCSV) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images=[]
measurements=[]

print("Reading images")
for line in lines:
    source_path = line[0]
    filename = source_path.split(FILEDELIM)[-1]
    current_path = IMAGEDIR+filename
    image = cv2.imread(current_path)
    images.append(image)
    images.append(cv2.flip(image,1))
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement*(-1.0))

print("Done")
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from myplainnet import SimpleNet
from mynvidianet import NvidiaNet
from keras.layers import Cropping2D

model = Sequential()
#model.add(Lambda(lambda x: x/255.0-0.5),input_shape=(160,320,3))) # Move this after crop
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

if(USENVIDIA) :
    NvidiaNet(model)
else :
    SimpleNet(model)


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True,nb_epoch=EPOCHS)

model.save('model.h5')
    
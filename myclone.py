import csv
import cv2
import numpy as np
MYCSV="record/driving_log.csv"
IMAGEDIR="record/IMG/"
FILEDELIM ="\\"
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
    measurement = float(line[3])
    measurements.append(measurement)

print("Done")
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True,nb_epoch=7)

model.save('model.h5')
    
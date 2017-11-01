import csv
import cv2
import numpy as np
RECORD_DIRS=["record","record_bridge","record_recover"]
CSVPATH="/driving_log.csv"
IMAGEDIR="/IMG/"
FILEDELIM ="\\"
USENVIDIA = True
USE_FLIP = True

CROP_TOP = 60
CROP_BOTTOM = 25

CORRECTION = 0.2

EPOCHS=5


images=[]
measurements=[]

for mdir in RECORD_DIRS:
    MYCSV = mdir + CSVPATH
    print("Reading images in "+MYCSV)
    lines = []
    with open(MYCSV) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)



  
    for line in lines:
        source_path = line[0]
        filename = source_path.split(FILEDELIM)[-1]
        current_path = mdir+IMAGEDIR+filename
        image = cv2.imread(current_path)
        images.append(image)
        if(USE_FLIP):
            images.append(cv2.flip(image,1))
        measurement = float(line[3])
        measurements.append(measurement)
        if(USE_FLIP):
            measurements.append(measurement*(-1.0))

print("Done :" +str(len(images)) + " images")
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from myplainnet import SimpleNet
from mynvidianet import NvidiaNet
from keras.layers import Cropping2D
from keras.layers import Lambda

model = Sequential()
#model.add(Lambda(lambda x: x/255.0-0.5),input_shape=(160,320,3))) # Move this after crop
model.add(Cropping2D(cropping=((CROP_TOP,CROP_BOTTOM), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

if(USENVIDIA) :
    NvidiaNet(model)
else :
    SimpleNet(model)


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True,nb_epoch=EPOCHS)

model.save('model.h5')
    
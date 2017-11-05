import csv
import cv2
import numpy as np
import sklearn

if (False) :
    RECORD_DIRS=["record","record_bridge","record_large","record_recover","record_track_2"]#,"record_off_track"]#]
else :
    RECORD_DIRS=["data"]

CSVPATH="/driving_log.csv"
IMAGEDIR="/IMG/"

FILEDELIM ="\\"
USENVIDIA = True
USE_FLIP = True
USE_SIDE_CAMS = True

CROP_TOP = 60
CROP_BOTTOM = 25

CORRECTION = 0.2

EPOCHS=5
BATCH_SIZE=32



def getImage(dirp, line, idx) :
    source_path = line[idx]
    #filename = source_path.split(FILEDELIM)[-1]
    #current_path =dirp+filename
    image = cv2.imread(source_path.strip())
    #Convert to RGB, because cv2 reads BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert image is not None, "Image <%s> returned None" % source_path
    return image

def addTrain(images,measurements,image,measurement) :
    images.append(image)
    measurements.append(measurement)
    if(USE_FLIP):
        images.append(cv2.flip(image,1))
        measurements.append(-measurement)    

csv_lines = []
for mdir in RECORD_DIRS:
    MYCSV = mdir + CSVPATH
    print("Reading images in "+MYCSV)
    with open(MYCSV) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            csv_lines.append(line)

# Split inspired from Udacity lesson
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(csv_lines, test_size=0.2)

print("Size of lines "+str(len(csv_lines)))
print("Train samples "+str(len(train_samples)))
print("Val samples "+str(len(validation_samples)))

def generator(samples, batch_size = BATCH_SIZE)  :
    num_samples = len(samples)
    while(True):
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            #print("Offset "+str(offset))
            images=[]
            measurements=[]

            for line in batch_samples:
                image = getImage(mdir+IMAGEDIR,line, 0)
                measurement = float(line[3])
                addTrain(images,measurements,image,measurement)
                if(USE_SIDE_CAMS):
                    image_left = getImage(mdir+IMAGEDIR,line, 1)
                    addTrain(images,measurements,image_left,measurement+CORRECTION)
                    image_right = getImage(mdir+IMAGEDIR,line, 2)
                    addTrain(images,measurements,image_right,measurement-CORRECTION)


            X_train = np.array(images)
           # print(len(images))
           # print("**********")
           # print(X_train.shape)
            y_train = np.array(measurements)
            # print(X_train)
            yield sklearn.utils.shuffle(X_train, y_train)
        

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

#Define generators for the train and validation set.
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True,nb_epoch=EPOCHS)

sample_factor = 1
if(USE_FLIP) :
    sample_factor = 2

if(USE_SIDE_CAMS) :
    sample_factor = sample_factor *3

model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*sample_factor, validation_data=validation_generator, nb_val_samples=len(validation_samples)*sample_factor, nb_epoch=3)

model.save('model.h5')
    
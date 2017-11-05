import csv
import cv2
import numpy as np
import sklearn

# A simple switch to toggle between Udacity provided data
# and own recordings. Code has been implemented so that
# data can be recorded into more than one directory for
# ease of use and comparability
#
if (False) :
    RECORD_DIRS=["record","record_bridge","record_large","record_recover","record_track_2"]#,"record_off_track"]#]
else :
    RECORD_DIRS=["data"]

# Some variables for the code that reads data from
# several directories.
CSVPATH="/driving_log.csv"
IMAGEDIR="/IMG/"
FILEDELIM ="\\"

# Flag to choose Nvidia NN or simple 1-layer
USENVIDIA = True

# Flag to toggle data augmentation of flipping image and
# angle
USE_FLIP = True

# Flag to toggle use of sidecams
USE_SIDE_CAMS = True
CORRECTION = 0.2

# Configuration of cropping dimensions
CROP_TOP = 60
CROP_BOTTOM = 25

# NN configuration
EPOCHS=5
BATCH_SIZE=32


# This reads an image based on a line in the csv. Idx
# gives column in csv to consider.
# Throws an assertion in case image reading failes.
# IMPORTANT: cv2 read BGR data, but drive.py works on
# RGB -> need to convert for training
#
def getImage(dirp, line, idx) :
    source_path = line[idx]
    #filename = source_path.split(FILEDELIM)[-1]
    #current_path =dirp+filename
    image = cv2.imread(source_path.strip())
    #Convert to RGB, because cv2 reads BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert image is not None, "Image <%s> returned None" % source_path
    return image

# Add training data to the set of data. Uses the
# USE_FLIP toggle to add a flipped image/angle if True
#
def addTrain(images,measurements,image,measurement) :
    images.append(image)
    measurements.append(measurement)
    if(USE_FLIP):
        images.append(cv2.flip(image,1))
        measurements.append(-measurement)    

# Read the lines from all the CSVs in all the directories
# that are configures as input
#
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

# Using a Python generator to provide data, since
# otherwise we get an OOM exception.
# Honors the flag of possible adding side came images
#
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

# Initial model, including image pre-processing. Note that the
# lambda is _after_ the cropping to reduce working on unnecessary data.
#
model = Sequential()
#model.add(Lambda(lambda x: x/255.0-0.5),input_shape=(160,320,3))) # Move this after crop
model.add(Cropping2D(cropping=((CROP_TOP,CROP_BOTTOM), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# The actual NN implementation is in mynvidianet.py, this is done to
# keep this code clean, and because I want to easily use a simpler NN for
# infrastructure tests
#
if(USENVIDIA) :
    NvidiaNet(model)
else :
    SimpleNet(model)

#Define generators for the train and validation set.
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True,nb_epoch=EPOCHS)

# For using the generator, we need to pre-calculate the number of images that
# we actually use. This is influenced by the data augmentation strategies being
# used, to the factor is calculated based on these.
#
sample_factor = 1
if(USE_FLIP) :
    sample_factor = 2

if(USE_SIDE_CAMS) :
    sample_factor = sample_factor *3

model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*sample_factor, validation_data=validation_generator, nb_val_samples=len(validation_samples)*sample_factor, nb_epoch=3)

model.save('model.h5')
    
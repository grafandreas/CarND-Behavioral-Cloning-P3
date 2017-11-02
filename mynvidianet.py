from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
def NvidiaNet(model) :
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))

    model.add(Flatten())
    model.add(Dense(100)) 
    model.add(Dropout(0.3))
    model.add(Dense(50)) 
    model.add(Dropout(0.3))
    model.add(Dense(10)) 
    model.add(Dropout(0.3))
    model.add(Dense(1)) 
    return model
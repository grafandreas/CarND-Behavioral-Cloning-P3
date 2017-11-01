from keras.models import Sequential
from keras.layers import Flatten, Dense

def SimpleNet(model) :
    model.add(Flatten(input_shape=(90,320,3)))
    model.add(Dense(1)) 
    return model
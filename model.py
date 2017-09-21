from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation


def init_model():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='valid', input_shape=(100, 100, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))

    model.add(Dense(1000))
    model.add(Activation('relu'))

    model.add(Dense(1000))
    model.add(Activation('relu'))

    model.add(Dense(1000))
    model.add(Activation('relu'))

    model.add(Dense(28))

    model.compile(loss='mse', optimizer='adam')
    return model

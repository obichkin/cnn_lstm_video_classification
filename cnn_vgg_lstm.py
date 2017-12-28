import time
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Flatten, Dense, LSTM, TimeDistributed
import keras


import cv2
import os
import numpy as np
import pandas as pd
from vgg_16_keras import VGG_16


video_path = "data-samples"
m = 256
batch_size = 32
cats = {"agitated": 0,
        "feeding": 1,
        "normal": 2,
        "obstructed": 3}
vgg16_weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
model_path = "my_model.h5"


def main():

    print("keras.__version__", keras.__version__)
    cnn_lstm_video_classification()



def cnn_lstm_video_classification():
    #model = load_vgg16_model_sequential()
    model = load_vgg16_model_api()

    model.fit_generator(
        generator=my_generator(batch_size),
        steps_per_epoch=16,
        epochs=1,
        verbose=2,
        validation_data=my_generator(batch_size),
        validation_steps=64
    )

    model.save(model_path)


def fit_model(model, test):

    X_train = np.zeros((0, 224, 224, 3))
    Y_train = np.zeros((0, 4), dtype=np.int)



    for dir in os.listdir(video_path):
        for file in [os.path.join(video_path, dir, f) for f in os.listdir(os.path.join(video_path, dir)) if f.endswith("normal-008.mp4")]:

            cap = cv2.VideoCapture(file)
            ret, frame = cap.read()
            width, height, channels = frame.shape
            side = min(width, height)

            x0 = int((width - side) / 2)
            x1 = int((width + side) / 2)
            y0 = int((height - side) / 2)
            y1 = int((height + side) / 2)

            i = 0
            x_train = np.zeros((m, 224, 224, 3))

            while (ret and i<m):
                ret, frame = cap.read()

                if(ret):
                    pic = cv2.resize(frame[x0:x1, y0:y1, :], (224, 224))
                    x_train[i] = pic
                    i += 1

                    #if (cv2.waitKey(1) & 0xFF == ord('q')):
                    #    break

            cap.release()
            cv2.destroyAllWindows()

            x_train = preprocess_input(x_train[:i])

            y_train = np.zeros((i,4), dtype=np.int)
            y_train[:, cats[dir]] = 1

            if(not test):
                print("fit", file)
                model.fit(
                    x=x_train,
                    y=y_train,
                    epochs=1
                )
            else:
                print("test", file)
                scores = model.evaluate(
                    x=x_train,
                    y=y_train
                )
                print(scores)

def my_generator(batch_size):
    i = 0
    x_train = np.zeros((batch_size, 224, 224, 3))
    y_train = np.zeros((batch_size, 4), dtype=np.int)

    (height, width, channels) = (1080, 1920, 3)
    side = min(height, width)

    x0 = int((width - side) / 2)
    x1 = int((width + side) / 2)
    y0 = int((height - side) / 2)
    y1 = int((height + side) / 2)

    for dir in os.listdir(video_path):
        for file in [os.path.join(video_path, dir, f) for f in os.listdir(os.path.join(video_path, dir)) if f.endswith(".mp4")]:

            #print(file)
            cap = cv2.VideoCapture(file)

            while True:
                ret, frame = cap.read()
                if(ret):
                    pic = cv2.resize(frame[x0:x1, y0:y1, :], (224, 224))

                    x_train[i] = pic
                    y_train[i, cats[dir]] = 1

                    i += 1

                    if(i >= batch_size):
                        x_train = preprocess_input(x_train[:i])
                        yield x_train, y_train
                        i=0
                        x_train = np.zeros((batch_size, 224, 224, 3))
                        y_train = np.zeros((batch_size, 4), dtype=np.int)
                else:
                    break
            cap.release()
            #cv2.destroyAllWindows()

def load_vgg16_model_sequential():

    if(os.path.isfile(model_path)):
        my_model = load_model(model_path)
    else:
        my_model = Sequential()

        my_model = load_model(vgg16_weights_path)
        my_model.summary()


#        my_model.load_weights(filepath=vgg16_weights_path, by_name=True)
#        my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



    return my_model

def load_vgg16_model_api():
    if(os.path.isfile(model_path)):
        my_model = load_model(model_path)
    else:

        inp = Input(shape=(224,224,3))
        vgg_16 = VGG16(include_top=False, weights='imagenet', input_tensor=inp)
        x = vgg_16.output

        for layer in vgg_16.layers:
            layer.trainable = False

        x = TimeDistributed(Flatten())(x)
        x = LSTM(32)(x)

        predictions = Dense(4, name="dense1")(x)

        my_model = Model(inputs=vgg_16.inputs, outputs=predictions)
        my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        #for layer in my_model.layers:
        #    print(layer.trainable, layer)

        return my_model

if __name__ == "__main__":
    main()



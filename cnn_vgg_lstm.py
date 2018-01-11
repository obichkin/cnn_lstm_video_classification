from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Flatten, Dense, LSTM, TimeDistributed
from keras.callbacks import ModelCheckpoint
#import keras

import timeit
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


video_path = "data-samples"
m = 256
batch_size = 16
cats = {"agitated": 0,
        "feeding": 1,
        "normal": 2,
        "obstructed": 3}

colors = [(255, 0, 0), (64, 255, 128), (0, 0, 255), (255, 255, 255)]

vgg16_weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
model_path = "my_model.h5"
output = "output.mp4"
videofiles_extension = "mp4"
output_shape = (640, 480)


def main():

    params = init()
    #cnn_lstm_video_classification(params)

    start = timeit.default_timer()

    validate_model(params)

    elapsed = timeit.default_timer() - start
    print(elapsed)

def validate_model(params):
    model = load_model_api()

    x0 = params['x0']
    x1 = params['x1']
    y0 = params['y0']
    y1 = params['y1']

    prob = pd.DataFrame(columns=['category', 'probability'])
    prob['category'] = cats.keys()

    x = np.zeros((1, 224, 224, 3), dtype=np.float64)

    fig = plt.figure()
    ax0 = fig.add_axes([0, 0, 1, 1])
    ax1 = fig.add_axes([0.2, 0.2, 0.2, 0.2])

    fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    out = cv2.VideoWriter('output.AVI', fourcc, 30, (640, 480))

    for dir in os.listdir(video_path):
        for file in [os.path.join(video_path, dir, f)
                     for f in os.listdir(os.path.join(video_path, dir))
                     if f.endswith(videofiles_extension)]:

            print(file)
            cap = cv2.VideoCapture(file)

            while True:
                ret, frame = cap.read()
                if (ret):
                    x[0] = cv2.resize(frame[x0:x1, y0:y1, :], (224, 224))
                    x = preprocess_input(x, mode='caffe', data_format='channels_last')
                    y_pred = model.predict(x)[0]
                    frame = cv2.resize(frame, output_shape)

                    #print(y_pred)
                    h=0
                    for category in cats.keys():
                        cv2.putText(frame, category, (50, 50 + h*15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color=colors[h%4])
                        cv2.line(frame, (200, 50 + h*15), (200+ int(y_pred[cats[category] ]*100), 50 + h*15), color=colors[h%4], thickness=7)
                        h += 1

                    #cv2.imshow("frame", frame)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break

                    out.write( cv2.resize(frame, output_shape) )
                else:
                    break

            cap.release()
            cv2.destroyAllWindows()
    out.release()


def cnn_lstm_video_classification(params):
    model = load_model_api()

    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        verbose=2,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1)
    callbacks_list = [checkpoint]


    model.fit_generator(
        generator=my_generator(batch_size, params),
        steps_per_epoch=32,
        epochs=50,
        verbose=2
        ,callbacks=callbacks_list
        #,validation_data=my_generator(batch_size, params)
        #,validation_steps=8
    )

    model.save(model_path)

def init():
    params = {}

    height = 1080
    width = 1920
    channels = 3
    side = 1080

    params['x0'] = int((width - side) / 2)
    params['x1'] = int((width + side) / 2)
    params['y0'] = int((height - side) / 2)
    params['y1'] = int((height + side) / 2)

    params['skip'] = np.random.randint(12)

    return params

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

def my_generator(batch_size, params):
    i = 0
    j = 0

    x_train = np.zeros((batch_size, 224, 224, 3))
    y_train = np.zeros((batch_size, 4), dtype=np.int)

    x0 = params['x0']
    x1 = params['x1']
    y0 = params['y0']
    y1 = params['y1']
    skip = params['skip']

    while True:
        for dir in os.listdir(video_path):
            for file in [os.path.join(video_path, dir, f) for f in os.listdir(os.path.join(video_path, dir)) if f.endswith(".mp4")]:

                if (j < skip):
                    j += 1
                    continue

                print(file)
                cap = cv2.VideoCapture(file)

                while True:
                    ret, frame = cap.read()
                    if(ret):
                        pic = cv2.resize(frame[x0:x1, y0:y1, :], (224, 224))

                        x_train[i] = pic
                        y_train[i, cats[dir]] = 1

                        i += 1

                        if(i >= batch_size):
                            x_train = preprocess_input(x_train[:i], mode='caffe', data_format='channels_last')
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

def load_model_api():
    if(os.path.isfile(model_path)):
        my_model = load_model(model_path)

    else:

        inp = Input(shape=(224,224,3))
        vgg_16 = VGG16(include_top=False, weights='imagenet', input_tensor=inp)
        x = vgg_16.output

        for layer in vgg_16.layers:
            layer.trainable = False

        x = TimeDistributed(Flatten())(x)
        x = LSTM(256, return_sequences=False, dropout=0.5)(x)
        predictions = Dense(4, name="dense1", activation="softmax")(x)

        my_model = Model(inputs=vgg_16.inputs, outputs=predictions)
        my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        #for layer in my_model.layers:
        #    print(layer.trainable, layer)

    return my_model

if __name__ == "__main__":
    main()



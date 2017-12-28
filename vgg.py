from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Input

import cv2
import os
import numpy as np
import pandas as pd

video_path = "data-samples"
m = 100  # number of frames per file
cats = eval(open("imagenet1000_clsid_to_human.txt").read()) #imagenet 1000 categories
nth_frame = 10 # sample each 10th frame of the video

def main():
    test_videos()


def test_videos():


    model = load_vgg16_model()
    Y_pred = np.zeros((0), dtype=np.int)

    for dir in os.listdir(video_path):
        for file in [os.path.join(video_path, dir, f) for f in os.listdir(os.path.join(video_path, dir)) if f.endswith(".mp4")]:
            y_pred = test_file(model, file)
            Y_pred = np.concatenate((Y_pred, y_pred))

    print(pd.Series(Y_pred).apply(lambda x: cats[x]).value_counts())



def test_file(model, file):
    print("scanning the file", file)
    cap = cv2.VideoCapture(file)
    ret, frame = cap.read()
    width, height, channels = frame.shape
    side = min(width, height)


    x0 = int((width-side)/2)
    x1 = int((width+side)/2)
    y0 = int((height-side)/2)
    y1 = int((height+side)/2)

    x_test = np.zeros([m, 224, 224, 3])
    i = 0
    j = 0
    ret = True

    while(ret and i<m):
        ret, frame = cap.read()


        if(ret and j%nth_frame==0):
            pic = cv2.resize( frame[x0:x1, y0:y1, :] , (224, 224))
            x_test[i] = pic
            i += 1

        j += 1
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break


    cap.release()
    cv2.destroyAllWindows()



    x_test = preprocess_input(x_test[:i])
    y_pred = np.argmax(model.predict(x_test), axis=1)
    print(y_pred)

    return y_pred


def load_vgg16_model():
    input = Input(shape=(224, 224, 3), name = "my_input")
    vgg16 = VGG16(include_top=True, weights='imagenet', input_tensor=input)
    #vgg16.summary()
    return vgg16

if __name__ == "__main__":
    main()
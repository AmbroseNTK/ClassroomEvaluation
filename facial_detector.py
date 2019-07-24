import cv2
import json
import os
import argparse
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import logger
import sys
import numpy as np

#ap = argparse.ArgumentParser()
#ap.add_argument("-s", "--session-id")

#args = vars(ap.parse_args())

session_id = sys.argv[1]

image_input = os.listdir("result/"+session_id+"/frames")
cascade_input = "opencv/data/haarcascades/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascade_input)

label_map = ["Angry", "Disgust", "Fear",
             "Happy", "Neutral", "Sad", "Surprise", ]
enable_vec = [1, 1, 1, 1, 1, 1, 1]

loaded_model = tf.keras.models.load_model(
    "facial_model.tf", custom_objects={'KerasLayer': hub.KerasLayer})

for img_name in image_input:
    image = cv2.imread("result/" + session_id + "/frames/" + img_name)
    print("Read "+img_name)
    image_origin = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    data = {}
    i = 0
    for (x, y, w, h) in faces:
        data[str(i)] = {
            "x": str(x),
            "y": str(y),
            "w": str(w),
            "h": str(h)
        }
        i += 1
    inference_x = []
    face_list = []
    for key in data:
        dat = data[key]
        x = int(dat.get('x'))
        y = int(dat.get('y'))
        w = int(dat.get('w'))
        h = int(dat.get('h'))
        crop_img = image_origin[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, dsize=(224, 224))
        face_list.append(resized_img)
        resized_img = np.expand_dims(resized_img, axis=0)
        inference_x.append(resized_img.astype('float32'))
    k = 0
    for input_x in inference_x:
        preds = loaded_model.predict(input_x/255)
        i = 0
        emotion = {}
        for pred in preds[0]:
            emotion[label_map[i]] = json.dumps(str(pred))
            #print("%s:%.4f" % (label_map[i], pred))
            i += 1
        data[str(k)]['emotion'] = emotion
        k += 1
    output = open("result/" + session_id +
                  "/facial/" + img_name + ".json", "w")
    json.dump(data, output)
    output.close()
    print("Facial: "+img_name+" done")

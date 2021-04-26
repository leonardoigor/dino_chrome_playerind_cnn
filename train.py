import keyboard
import uuid
import time
from PIL import Image
from mss import mss
import cv2
import numpy as np


color = (0, 255, 0) # bounding box color.

# This defines the area on the screen.
mon = {"top":300,
       "left":970,
       "width":800,
       "height":445}
sct = mss()

i = 0

def record_screen(record_id, key):
    global i # I will use this i inside and outside of the function. (I have an other i)
    print(f"{key}, {i}") #key: char of keyboard, i: num of press for char

    sct.get_pixels(mon)
    im = Image.frombytes('RGB', (sct.width, sct.height), sct.image )
    im.save(f"./data/img/{key}_{record_id}_{i}.png")



is_exit = False

def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey("esc", exit)


while True:
    break
    record_id = uuid.uuid4()

    if is_exit: break

    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id=record_id, key="up")
            time.sleep(0.1)

        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id=record_id, key="down")
            time.sleep(0.1)

        elif keyboard.is_pressed("right"):
            record_screen(record_id=record_id, key="right")
            time.sleep(0.1)

    except RuntimeError: continue














import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

imgs = glob.glob("./data/img/*.png")



width = 40
height = 10












X = [] # images ("cactus", "bird")
y = [] # labels ("up", "right", "down")

for img in imgs:

    filename = os.path.basename(img)
    label = filename.split("_")[0] # up, down, right
    im = np.array(Image.open(img).convert("L").resize((width, height)))
    im = im/255 # normalization
    
    cv2.imshow('teste',im)
    cv2.waitKey(1)
    X.append(im)
    y.append(label)







X = np.array(X)
X = X.reshape(X.shape[0], width, height, 1)


















def one_hot_labels(values):

    # Label Encoding -> One Hot Encoding

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded

# One Hote Encoding
Y = one_hot_labels(y)









train_X, test_X, train_y, test_y = train_test_split(X, Y , test_size = 0.25, random_state = 2)
print(train_X.shape)






# CNN Model
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (width, height, 1)))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(3, activation = "softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="Adam",
              metrics=["acc"])

# training            
model.fit(train_X, train_y, epochs = 100, batch_size = 64) 




score_train = model.evaluate(train_X, train_y)
print("Training Score: %", score_train[1]*100)

score_test = model.evaluate(test_X, test_y)
print("Test Score: %", score_test[1]*100)





# save weights
open("trex_model.json","w").write(model.to_json())
model.save_weights("./trex_weight.h5")











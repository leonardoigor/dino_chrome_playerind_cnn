from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
import os
from mss import mss
import cv2

import time


# This defines the area on the screen.
mon = {"top":300,
       "left":970,
       "width":800,
       "height":445}
sct = mss()


# size of images
width = 40
height = 10



# load model
model = model_from_json(open("./trex_model.json", "r").read())
model.load_weights("./trex_weight.h5")



#down:0, right:1, up:2
labels = ["Down", "Right", "Up"]





framerate_time = time.time()
counter = 0
i = 0
delay = 0.4
key_down_pressed = False
while True:
    start_time = time.time() # start time of the loop

    sct.get_pixels(mon)
    im = Image.frombytes( 'RGB', (sct.width, sct.height), sct.image )
    im2 = np.array(im.convert("L").resize((width, height)))
    #im2 = cv2.resize(im2, (400, 100), interpolation=cv2.INTER_AREA)
    cv2.imshow('aa',cv2.resize(im2,(800,400)))
    im2 = im2 / 255  # normalization
    
    X = np.array([im2])
    X = X.reshape(X.shape[0], width, height, 1)
   
    r = model.predict(X)

    result = np.argmax(r)
    if result == 0: #down: 0

        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed = True
        print('dowm')

    elif result == 2: #up: 2

        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
            time.sleep(delay)

        keyboard.press(keyboard.KEY_UP)
        print('up')

        #if i < 1500:
         #   time.sleep(0.3)

        #elif 1500 < i and i < 5000:
         #   time.sleep(0.2)

        #else:
         #   time.sleep(0.17)

        #keyboard.press(keyboard.KEY_DOWN)
        #keyboard.release(keyboard.KEY_DOWN)

    counter += 1
    '''
    if (time.time() - framerate_time) > 1:

        counter = 0
        framerate_time = time.time()

        if i <= 1500:
            delay -= 0.003

        else:
            delay -= 0.005

        if delay < 0:
            delay = 0

        '''
    i += 1
    print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
    #print(result)
    if cv2.waitKey ( 1 ) & 0xff == ord( 'q' ) :
        break
cv2.destroyAllWindows()

































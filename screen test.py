# import required libraries
from vidgear.gears import ScreenGear
import cv2
import time

# define dimensions of screen w.r.t to given monitor to be captured
options = {"top":300,
       "left":970,
       "width":800,
       "height":445}

# open video stream with defined parameters
stream = ScreenGear(monitor=2, logging=True, **options).start()

# loop over
while True:
    start_time = time.time() # start time of the loop

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break


    # {do something with the frame here}


    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()
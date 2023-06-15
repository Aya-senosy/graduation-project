import cv2
import pyttsx3
engine = pyttsx3.init()
engine.say("color detection has been activated")
img = cv2.imread(r'C:\Users\ayase\OneDrive\Desktop\graduation project\3.png')
# img.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
# img.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# while True:
#     _, frame = cap.read()
hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
height, width, _ = img.shape

cx = int(width / 2)
cy = int(height / 2)

# Pick pixel value
pixel_center = hsv_frame[cy, cx]
hue_value = pixel_center[0]
saturation = pixel_center[1]
value = pixel_center[2]


engine.setProperty( "rate", 200 )
engine.setProperty( "volume", 1.0 )
color = "Undefined"
if saturation < 20:
        if value > 200:
            color = engine.say("White")
        else:
            color = engine.say("White")
elif hue_value < 5 or hue_value > 175:
        color = engine.say("Red")
elif hue_value < 22:
        color = engine.say("Orange")
elif hue_value < 33:
        color = engine.say("Yellow")
elif hue_value < 78:
        color = engine.say("Green")
elif hue_value < 130:
        color = engine.say("Black")
elif hue_value < 170:
        color = engine.say("Magenta")
else:
        color = "Purple"

# Check for pink
if color == "Magenta" and value > 200:
        color = engine.say("Pink")
        
# Check for maroon
if color == "Red" and value < 50:
        color = engine.say("Maroon")

# Check for brown
if color == "Orange" and value < 100:
        color = engine.say("brown")

pixel_center_bgr = img[cy, cx]
b, g, r = int(pixel_center_bgr[0]), int(pixel_center_bgr[1]), int(pixel_center_bgr[2])
cv2.putText(img, color, (10, 50), 5, 3, (b, g, r), 4)
cv2.circle(img, (cx, cy), 25, (255, 0, 0), 5)
if(color == "Undefined") is True:
        engine.say("nothing detected, please try again")

    
engine.runAndWait()

# cv2.waitKey(1)

# cv2.imshow('Image', img)
# cv2.destroyAllWindows()

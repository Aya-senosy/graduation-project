import cv2
import numpy as np
from keras.models import model_from_json
import pyttsx3
engine = pyttsx3.init()
engine.say("emotion detection has been activated")
emotion_dict = {0: "Angry", 1: "Neutral", 2: "Neutral", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open(r'C:\Users\ayase\OneDrive\Desktop\graduation project\emotion recognition\emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights(r"C:\Users\ayase\OneDrive\Desktop\graduation project\emotion recognition\emotion_model")

img = cv2.imread(r'C:\Users\ayase\OneDrive\Desktop\graduation project\5.jpg')

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
#cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")

# while True:
    # Find haar cascade to draw bounding box around face
# ret, frame = cap.read()
frame = cv2.resize(img, (1000, 700))

face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        print(emotion_dict[maxindex])
        engine.say(emotion_dict[maxindex])
if (len(num_faces)<1):
        engine.say("nothing detected, please try again")

engine.runAndWait()

cv2.imshow('Emotion Detection', frame)

cv2.destroyAllWindows()

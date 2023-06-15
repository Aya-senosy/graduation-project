import cv2
from PIL import Image
from pytesseract import pytesseract
import pyttsx3
engine = pyttsx3.init()
def AI_speak(label):
    engine.setProperty("rate", 170)
    engine.say(label)
    engine.runAndWait()
AI_speak("text recognition has been activated")
def tesseract():

 img = cv2.imread(r'C:\Users\ayase\OneDrive\Desktop\graduation project\10.png')

 path_to_tesseract = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
 pytesseract.tesseract_cmd = path_to_tesseract
 text = pytesseract.image_to_string(Image.fromarray(img))
 AI_speak(text)
 #-------------------------------
 i=0
 for n in text:
     if(n =="\n" or n==" "):
         continue
     else:
         i=i+1

 if(i<1):
     AI_speak("nothing detected, please try again")

tesseract()



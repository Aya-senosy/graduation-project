
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyttsx3
# %matplotlib inline
from glob import glob
import os
from os import listdir
from tqdm.notebook import tqdm
tqdm.pandas()
import cv2, warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Add, Dropout, LSTM, TimeDistributed, Embedding, RepeatVector, Concatenate, Bidirectional, Convolution2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
engine = pyttsx3.init()
def AI_speak(label):
    engine.setProperty("rate", 170)
    engine.say(label)
    engine.runAndWait()
AI_speak("image captioning has been activated")

# get the path/directory
folder_dir = "images"
images=[]
for image in os.listdir(folder_dir):
    # if (images.endswith(".png")):
        images.append(image)


captions = open('captions.txt','rb').read().decode('utf-8').split('\n')

inception_model = ResNet50(include_top=True)
# inception_model.summary()

last = inception_model.layers[-2].output # Output of the penultimate layer of ResNet model
model = Model(inputs=inception_model.input,outputs=last)
# model.summary()

"""## Extracting features from images"""

img_features = {}
count = 0

for img_path in tqdm(images):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224)) # ResNet model requires images of dimensions (224,224,3)
    img = img.reshape(1,224,224,3) # Reshaping image to the dimensions of a single image
    features = model.predict(img).reshape(2048,) # Feature extraction from images
    img_name = img_path.split('/')[-1] # Extracting image name
    img_features[img_name] = features
    count += 1
    # Fetching the features of only 1500 images as using more than 1500 images leads to overloading memory issues
#     if count == 1500:
#         break
#     if count % 50 == 0:
#         print(count)

"""## Preprocessing the captions text"""

captions = captions[1:]

captions[8].split(',')[1]

captions_dict = {}

for cap in captions:
    try:
        img_name = cap.split(',')[0]
        caption = cap.split(',')[1]
        # Each image has 5 captions
        if img_name in img_features:
            if img_name not in captions_dict:
                captions_dict[img_name] = [caption] # Storing the first caption
            else:
                captions_dict[img_name].append(caption) # Adding the remaining captions
    except:
        break

def text_preprocess(text):
    modified_text = text.lower() # Converting text to lowercase
    modified_text = 'startofseq ' + modified_text + ' endofseq' # Appending the special tokens at the beginning and ending of text
    return modified_text

# Storing the preprocessed text within the captions dictionary
for key, val in captions_dict.items():
    for item in val:
        captions_dict[key][val.index(item)] = text_preprocess(item)

"""## Creating vocabulary of the entire text corpus"""

count_words = dict()
cnt = 1

for key, val in captions_dict.items(): # Iterating through all images with keys as images and their values as 5 captions
    for item in val: # Iterating through all captions for each image
        for word in item.split(): # Iterating through all words in each caption
            if word not in count_words:
                count_words[word] = cnt
                cnt += 1

# Encoding the text by assigning each word to its corresponding index in the vocabulary i.e. count_words dictionary
for key, val in captions_dict.items():
    for caption in val:
        encoded = []
        for word in caption.split():
            encoded.append(count_words[word])
        captions_dict[key][val.index(caption)] = encoded

# Determining the maximum possible length of text within the entire captions text corpus
max_len = -1

for key, value in captions_dict.items():
    for caption in value:
        if max_len < len(caption):
            max_len = len(caption)

vocab_size = len(count_words) # Vocab size is the total number of words present in count_words dictionary

"""## Building a custom generator function to generate input image features, previously generated text and the text to be generated as output"""

def generator(img,caption):
    n_samples = 0
    X = []
    y_input = []
    y_output = []

    for key, val in caption.items():
        for item in val:
            for i in range(1,len(item)):
                X.append(img[key]) # Appending the input image features
                input_seq = [item[:i]] # Previously generated text to be used as input to predict the next word
                output_seq = item[i] # The next word to be predicted as output
                # Padding encoded text sequences to the maximum length
                input_seq = pad_sequences(input_seq,maxlen=max_len,padding='post',truncating='post')[0]
                # One Hot encoding the output sequence with vocabulary size as the total no. of classes
                output_seq = to_categorical([output_seq],num_classes=vocab_size+1)[0]
                y_input.append(input_seq)
                y_output.append(output_seq)

    return X, y_input, y_output

X, y_in, y_out = generator(img_features,captions_dict)

# Converting input and output into Numpy arrays for faster processing
X = np.array(X)
y_in = np.array(y_in,dtype='float64')
y_out = np.array(y_out,dtype='float64')

X.shape, y_in.shape, y_out.shape

"""## Establishing the model architecture"""

embedding_len = 128
MAX_LEN = max_len
vocab_size = len(count_words)

# Model for image feature extraction
img_model = Sequential()
img_model.add(Dense(embedding_len,input_shape=(2048,),activation='relu'))
img_model.add(RepeatVector(MAX_LEN))

img_model.summary()

# Model for generating captions from image features
captions_model = Sequential()
captions_model.add(Embedding(input_dim=vocab_size+1,output_dim=embedding_len,input_length=MAX_LEN))
captions_model.add(LSTM(256,return_sequences=True))
captions_model.add(TimeDistributed(Dense(embedding_len)))

captions_model.summary()

# Concatenating the outputs of image and caption models
concat_output = Concatenate()([img_model.output,captions_model.output])
# First LSTM Layer
output = LSTM(units=128,return_sequences=True)(concat_output)
# Second LSTM Layer
output = LSTM(units=512,return_sequences=False)(output)
# Output Layer
output = Dense(units=vocab_size+1,activation='softmax')(output)
# Creating the final model
final_model = Model(inputs=[img_model.input,captions_model.input],outputs=output)
final_model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics='accuracy')
final_model.summary()

"""## Visualizing the model architecture

## Model training
"""

from tensorflow import keras
final_model=keras.models.load_model("image_caption_generator.h5")

# Creating an inverse dictionary with reverse key-value pairs
inverse_dict = {val: key for key,val in count_words.items()}

"""## Saving the final trained model and the vocabulary dictionary"""

np.load('vocab.npy', allow_pickle=True)

"""## Generating sample predictions"""

def getImg(path):
    test_img = cv2.imread(path)
    test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img,(224,224))
    test_img = np.reshape(test_img,(1,224,224,3))
    return test_img

test_img_path = "1.jpg"
test_feature = model.predict(getImg(test_img_path)).reshape(1,2048)
test_img = cv2.imread(test_img_path)
test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
pred_text = ['startofseq']
count = 0
caption = '' # Stores the predicted captions text

while count < 25:
  count += 1
# Encoding the captions text with numbers
  encoded = []

  for i in pred_text:
           encoded.append(count_words[i])
  encoded = [encoded]
  # Padding the encoded text sequences to maximum length
  encoded = pad_sequences(encoded,maxlen=MAX_LEN,padding='post',truncating='post')
  pred_idx = np.argmax(final_model.predict([test_feature,encoded])) # Fetching the predicted word index having the maximum probability of occurrence
  sampled_word = inverse_dict[pred_idx] # Extracting the predicted word by its respective index
  # Checking for ending of the sequence
  if sampled_word == 'endofseq':
                  break
  caption = caption + ' ' + sampled_word
  pred_text.append(sampled_word)
AI_speak(caption)
plt.figure(figsize=(5,5))
plt.imshow(test_img)
plt.xlabel(caption)


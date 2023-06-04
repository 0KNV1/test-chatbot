import numpy as np
import random
import json
import pandas as pd

# Import Libraries
import json
import nltk
import time
import random
import string
import pickle
import numpy as np
import pandas as pd
from io import BytesIO
import tensorflow as tf
import IPython.display as ipd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Flatten, Dense
# Importing the dataset
with open('dedecorins.json') as content:
  data1 = json.load(content)

# Mendapatkan semua data ke dalam list
tags = [] # data tag
inputs = [] # data input atau pattern
responses = {} # data respon
words = [] # Data kata 
classes = [] # Data Kelas atau Tag
documents = [] # Data Kalimat Dokumen
ignore_words = ['?', '!'] # Mengabaikan tanda spesial karakter


for intent in data1['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['patterns']:
    inputs.append(lines)
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
      w = nltk.word_tokenize(pattern)
      words.extend(w)
      documents.append((w, intent['tag']))
      # add to our classes list
      if intent['tag'] not in classes:
        classes.append(intent['tag'])
        

# Konversi data json ke dalam dataframe
data = pd.DataFrame({"patterns":inputs, "tags":tags})
data

# Removing Punctuations (Menghilangkan Punktuasi)
data['patterns'] = data['patterns'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['patterns'] = data['patterns'].apply(lambda wrd: ''.join(wrd))
data

lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

print(len(words), "unique lemmatized words", words)

# sort classes
classes = sorted(list(set(classes)))
print(len(classes), "classes", classes)

# documents = combination between patterns and intents
print(len(documents), "documents")

# Tokenize the data (Tokenisasi Data)
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['patterns'])
train = tokenizer.texts_to_sequences(data['patterns'])
train

# Apply padding 
x_train = pad_sequences(train)

# Encoding the outputs 
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])
# le.fit(y_train)


print(x_train) # Padding Sequences
print(y_train) #Label Encodings

"""Tokenizer pada Tensorflow memberikan token unik untuk setiap kata yang berbeda. Dan juga padding dilakukan untuk mendapatkan semua data dengan panjang yang sama sehingga dapat mengirimkannya ke lapisan atau layer RNN. variabel target juga dikodekan menjadi nilai desimal.

# **Input Length, Output Length and Vocabulary**
"""

# input length
input_shape = x_train.shape[1]
print("input shape",input_shape)

vector = tokenizer.texts_to_sequences(data)
vector = np.array(vector).reshape(-1)
vector = pad_sequences([vector], input_shape)

# define vocabulary
vocabulary = len(tokenizer.word_index)
print("number of unique words : ", vocabulary)

# output length
output_length = le.classes_.shape[0]
print("output length: ", output_length)





"""**Input length** dan **output length** terlihat sangat jelas hasilnya. Mereka adalah untuk bentuk input dan bentuk output dari jaringan syaraf pada algoritma Neural Network.

**Vocabulary Size** adalah untuk lapisan penyematan untuk membuat representasi vektor unik untuk setiap kata.

# **Save Model Words & Classes**
"""

pickle.dump(words, open('model/words.pkl','wb'))
pickle.dump(classes, open('model/classes.pkl','wb'))
pickle.dump(tokenizer, open('model/tokenizer.pkl','wb'))
pickle.dump(le, open('model/label_encoder.pkl','wb'))

"""# **Modeling**"""

# Creating the model (Membuat Modeling)
i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,20)(i) # Layer Embedding
x = LSTM(20, return_sequences=True, recurrent_dropout=0.2)(x)
x = LSTM(20, return_sequences=True, recurrent_dropout=0.2)(x) # Layer Long Short Term Memory
x = Flatten()(x) # Layer Flatten
x = Dense(output_length, activation="softmax")(x) # Layer Dense
model  = Model(i,x)

# Compiling the model (Kompilasi Model)
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# Training the model (Latih Model Data)
train = model.fit(x_train, y_train, epochs=200)

# Plotting model Accuracy and Loss (Visualisasi Plot Hasil Akurasi dan Loss)
# Plot Akurasi
# plt.figure(figsize=(14, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train.history['accuracy'],label='Training Set Accuracy')
# plt.legend(loc='lower right')
# plt.title('Accuracy')
# # Plot Loss
# plt.subplot(1, 2, 2)
# plt.plot(train.history['loss'],label='Training Set Loss')
# plt.legend(loc='upper right')
# plt.title('Loss')
# plt.show()

model.save('model/chat_model.h5')
# Membuat Input Chat
# while True:
#   texts_p = []
#   prediction_input = input('Kamu : ')
  
#   # Menghapus punktuasi dan konversi ke huruf kecil
#   prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
#   prediction_input = ''.join(prediction_input)
#   texts_p.append(prediction_input)

#   vector = tokenizer.texts_to_sequences(texts_p)
#   vector = np.array(vector).reshape(-1)
#   vector = pad_sequences([vector], input_shape)

#   output = model.predict(vector)
#   output = output.argmax()
  # Menemukan respon sesuai data tag dan memainkan voice bot
  
  # response_tag = le.inverse_transform([output])[0]
  # print("Dedecorins : ", random.choice(responses[response_tag]))
  # #tts = gTTS(random.choice(responses[response_tag]), lang='id')
  # #tts.save('Dedecorins.wav')
  # #time.sleep(0.08)
  # #ipd.display(ipd.Audio('Dedecorins.wav', autoplay=False))
  # print("="*60 + "\n")
  # tag=[]
  # if response_tag == "goodbye":
  #   break
  # if response_tag == "abc":
  #   print("Dedecorins : ", "Maaf saya tidak mengetahui pertanyaan anda")





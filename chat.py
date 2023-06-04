import json
import random
import nltk
import string
import numpy as np
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
global responses, lemmatizer, tokenizer, le, model, input_shape
input_shape = 20
tags = [] # data tag
inputs = [] # data input atau pattern
responses = {} # data respon
words = [] # Data kata 
documents = [] # Data Kalimat Dokumen
classes = [] # Data Kelas atau Tag

# import dataset answer
def load_response():
    global responses
    responses = {}
    with open('dedecorins.json') as content:
        data = json.load(content)
    for intent in data['intents']:
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
  
# import model dan download nltk file
def preparation():
    load_response()
    global lemmatizer, tokenizer, le, model
    le = preprocessing.LabelEncoder()
    pickle.dump(le, open('model/label_encoder.pkl', 'wb'))

    le = preprocessing.LabelEncoder()
    tokenizer = pickle.load(open('model/tokenizer.pkl', 'rb'))
    le = pickle.load(open('model/label_encoder.pkl', 'rb'))
    model = keras.models.load_model('model/chat_model.h5')
    lemmatizer = WordNetLemmatizer()
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# def lemmatization(text):
#     word_list = nltk.word_tokenize(text)
#     print(word_list)
#     lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
#     print(lemmatized_output)
#     return lemmatized_output
def remove_punctuation(text):
    texts_p = []
    text = [letters.lower() for letters in text if letters not in string.punctuation]
    text = ''.join(text)
    texts_p.append(text)
    return texts_p

# mengubah text menjadi vector
def vectorization(texts_p):
    vector = tokenizer.texts_to_sequences(texts_p)
    vector = np.array(vector).reshape(-1)
    vector = pad_sequences([vector], input_shape)
    return vector

# klasifikasi pertanyaan user
def predict(vector):
    output = model.predict(vector).astype(int).ravel()
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    return response_tag

# menghasilkan jawaban berdasarkan pertanyaan user
def generate_response(text):
    texts_p = remove_punctuation(text)
    vector = vectorization(texts_p)
    response_tag = predict(vector)
    answer = random.choice(responses[response_tag])
    return answer
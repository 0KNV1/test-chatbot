import pickle

le = pickle.load(open("model/label_encoder.pkl", 'rb'))
print(le)
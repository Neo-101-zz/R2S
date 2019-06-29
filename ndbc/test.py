import pickle

pickle_file = open('./pickle/46001.pkl', 'rb')
windsat01 = pickle.load(pickle_file)
print(windsat01[0:10])
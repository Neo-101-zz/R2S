import pickle

pickle_file = open('./pickle/ascat_01_data.pkl', 'rb')
windsat01 = pickle.load(pickle_file)
print(len(windsat01[1]))
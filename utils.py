import gzip
import pickle

def save_obj(obj,path):
    with gzip.open(path,'w') as f:
        pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with gzip.open(path,'r') as f:
        obj = pickle.load(f)
    return obj
import pickle as pkl
from os.path import join as pjoin, exists as pexists
    

def load_data(data_name, data_path = './node/data/data_dict/'):
    file_path = pjoin(data_path, data_name + '.pickle')
    
    with open(file_path, 'rb') as file:
        # Deserialize and load the object from the file
        ds_pickled = pkl.load(file)
        
    return ds_pickled
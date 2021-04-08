import pickle

def load_pickle_file(path):
    with open(path, 'rb') as f:
        loaded_file = pickle.load(f)

    return loaded_file

def save_to_pickle(obj, save_to, description=''):
    with open(save_to, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Saved {description} to {save_to}')

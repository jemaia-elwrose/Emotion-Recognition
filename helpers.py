import os
import time
import math
import torch

def time_since(since):
    """
    Function to measure time between to operation.

    Args:
        since: start time

    Returns:
        formated string representing minutes and seconds
        
    """

    now = time.time()
    seconds = now - since
    minutes = math.floor(seconds / 60)
    seconds -= minutes * 60

    return f'{minutes}m {seconds:.4f}s'


def save_model(model, path, save_name):
    """
    Save the model at given path.

    Args:
        model: model's weight to saved
        path: path to save model
        save_name: name of file to be saved

    Returns:
        None
        
    """

    if not os.path.isdir(path):
        os.makedirs(path)

    model_path = path + save_name
    torch.save(model.state_dict(), model_path)


def load_model(path, save_name):
    """
    Function to load the pretrained weights.

    Args:
        path: path where weights are
        save_name: name of saved file

    Returns:
        state_dict: pretrained weights
        
    """

    model_path = path + save_name
    state_dict = torch.load(model_path, map_location='cuda' 
                            if torch.cuda.is_available() else 'cpu')

    return state_dict

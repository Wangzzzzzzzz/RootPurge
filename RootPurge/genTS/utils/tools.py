from logging import basicConfig
import matplotlib.pyplot as plt
import torch
import os
import shutil
import inspect
import numpy as np

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2, alpha=0.5)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2, alpha=0.5)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def save_copy_of_files(checkpoint_callback):
    """Copy from TSLANet"""
    # Get the frame of the caller of this function
    caller_frame = inspect.currentframe().f_back

    # Get the filename of the caller
    caller_filename = caller_frame.f_globals["__file__"]

    # Get the absolute path of the caller script
    caller_script_path = os.path.abspath(caller_filename)

    # Destination directory (PyTorch Lightning saving directory)
    destination_directory = checkpoint_callback.dirpath

    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # Copy the caller script to the destination directory
    shutil.copy(caller_script_path, destination_directory)

def str2bool(v):
    """Copy from TSLANet"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')



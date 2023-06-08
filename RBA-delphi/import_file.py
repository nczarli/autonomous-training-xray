"""
**********************************************************************************
 * Autonomous Training in X-Ray Imaging Systems
 * 
 * Training a deep learning model based on noisy labels from a rule based algorithm.
 * 
 * Copyright 2023 Nikodem Czarlinski
 * 
 * Licensed under the Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0)
 * (the "License"); you may not use this file except in compliance with 
 * the License. You may obtain a copy of the License at
 * 
 *     https://creativecommons.org/licenses/by-nc/3.0/
 * 
**********************************************************************************
"""

"""
This file imports all necessary modules and packages for inference.

Typical usage example:
    # Run the file in Delphi PythonEngine

    procedure InitPythonEngine stdcall;
    begin
        gEngine := TPythonEngine.Create(nil);
        gEngine.AutoFinalize := False;
        gEngine.UseLastKnownVersion := False;
        gEngine.RegVersion := '3.9';  //<-- Use the same version as the python 3.x your main program uses
        gEngine.APIVersion := 1013;
        gEngine.DllName := 'python39.dll';
        gEngine.LoadDll;
        gEngine.ExecFile('C:\XX\import_file.py');
        OutputDebugStringA('Finished Initialisation');
    end;
"""


import cv2                          # Import OpenCV for image processing
import glob                         # Import glob for file handling
import json                         # Import JSON for working with JSON files
import logging                      # Import logging for logging purposes
import matplotlib.pyplot as plt     # Import matplotlib for plotting
import numpy as np                  # Import NumPy for numerical computations
import os                           # Import os for operating system related functionalities
import pandas as pd                 # Import pandas for data manipulation and analysis
import pickle                       # Import pickle for object serialization
import re                           # Import re for regular expressions
import seaborn as sn                # Import seaborn for statistical visualization
import sys                          # Import sys for system-specific parameters and functions
import threading                    # Import threading for creating and managing threads
import time                         # Import time for time-related functions
import torch                        # Import PyTorch for deep learning
import torchvision.transforms as transforms  # Import torchvision for image transformations
import warnings                     # Import warnings to handle warnings gracefully

from scipy import ndimage           # Import ndimage from SciPy for image processing
from skimage import exposure        # Import exposure from scikit-image for image exposure adjustment
from skimage.feature import peak_local_max  # Import peak_local_max from scikit-image for peak detection
from skimage.segmentation import watershed  # Import watershed from scikit-image for image segmentation
from sklearn.mixture import GaussianMixture  # Import GaussianMixture from scikit-learn for Gaussian mixture models
from tqdm import tqdm               # Import tqdm for progress bar visualization

from tkinter import *               # Import tkinter for GUI development
from tkinter import filedialog      # Import filedialog from tkinter for file dialog window

# Append custom module path
sys.path.append('C:\\Users\\Nikodem\\Desktop\\Code\\SSR_BMVC2022')  # Add path for custom module

# Import custom modules
from models.preresnet import PreResNet18  # Import PreResNet18 model from models.preresnet
from utils import *                  # Import utility functions from utils
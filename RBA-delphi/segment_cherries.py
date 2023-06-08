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


import cv2  # Import OpenCV for image processing
import glob  # Import glob for file handling
import json  # Import JSON for working with JSON files
import logging  # Import logging for logging purposes
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import numpy as np  # Import NumPy for numerical computations
import os  # Import os for operating system related functionalities
import pandas as pd  # Import pandas for data manipulation and analysis
import pickle  # Import pickle for object serialization
import re  # Import re for regular expressions
import seaborn as sn  # Import seaborn for statistical visualization
import sys  # Import sys for system-specific parameters and functions
import threading  # Import threading for creating and managing threads
import time  # Import time for time-related functions
import torch  # Import PyTorch for deep learning
import torchvision.transforms as transforms  # Import torchvision for image transformations
import warnings  # Import warnings to handle warnings gracefully

from scipy import ndimage  # Import ndimage from SciPy for image processing
from skimage import (
    exposure,
)  # Import exposure from scikit-image for image exposure adjustment
from skimage.feature import (
    peak_local_max,
)  # Import peak_local_max from scikit-image for peak detection
from skimage.segmentation import (
    watershed,
)  # Import watershed from scikit-image for image segmentation
from sklearn.mixture import (
    GaussianMixture,
)  # Import GaussianMixture from scikit-learn for Gaussian mixture models
from tqdm import tqdm  # Import tqdm for progress bar visualization

from tkinter import *  # Import tkinter for GUI development
from tkinter import filedialog  # Import filedialog from tkinter for file dialog window

# Append custom module path
sys.path.append(
    "C:\\Users\\Nikodem\\Desktop\\Code\\SSR_BMVC2022"
)  # Add path for custom module

# Import custom modules
from models.preresnet import (
    PreResNet18,
)  # Import PreResNet18 model from models.preresnet
from utils import *  # Import utility functions from utils


warnings.filterwarnings("ignore")
print("[info] Import Successful")


def load_checkpoint(checkpoint_path):
    logging.debug("Loading checkpoint %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    return checkpoint


class CherryImage:
    """Class for image processing"""

    def __init__(self, image):
        self.original_image = image
        self.image = image
        self.small_holes_filled = None
        self.contour_image = image
        self.for_comparison = image
        self.n = 0

    def show(self, window_name):
        """
        Show the image in a window.
        
        Parameters
        ----------
        window_name : str
            Name of the window.
                
        Returns
        -------
        None.
        """
        cv2.imshow(window_name, self.image)

    def save(self, basepath, filename):
        pass

    def resize(self, x_stretch_factor, y_stretch_factor):
        """
        Resize the image.

        Parameters
        ----------
        x_stretch_factor : float
            Stretch factor in x direction.
        y_stretch_factor : float
            Stretch factor in y direction.  

        Returns
        -------
        None.
        """
        self.image = cv2.resize(
            self.image, (0, 0), fx=x_stretch_factor, fy=y_stretch_factor
        )
        self.resize = self.image.copy()

    def mean_shift_filtering(self, spatial_radius: float, color_radius: float):
        """
        performs pyramid mean shift filtering

        Parameters
        ----------
        spatial_radius : float
            spatial window radius
        color_radius : float
            color window radius

        Returns
        -------
        None.
        """

        self.image = cv2.pyrMeanShiftFiltering(self.image, spatial_radius, color_radius)
        self.mean_shift_filtering = self.image.copy()

    def closing(self, kernel_size: int, save_to_small_holes_filled: bool = False):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if save_to_small_holes_filled:
            self.small_holes_filled = cv2.morphologyEx(
                self.image, cv2.MORPH_CLOSE, kernel
            )
        else:
            self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
            self.closing = self.image.copy()

    def gray(self):
        """
        Convert the image to grayscale.

        The number of channels is reduced to 1 from 3 RGB channels.

        Returns
        -------
        None.
        """
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gray = self.image.copy()

    def blur(self):
        """
        Blur the image.
        
        Returns
        -------
        None.
        """
        self.image = cv2.GaussianBlur(self.image, (5, 5), sigmaX=10)
        self.blur = self.image.copy()

    def threshold(self):
        """
        Threshold the image.
        
        Returns
        -------
        None.
        """
        self.image = cv2.threshold(
            self.image, 0, 256, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]
        self.thresholded = self.image.copy()

    def erosion(self, kernel_size: int):
        """
        Erode the image.

        Parameters
        ----------
        kernel_size : int
            Size of the kernel.

        Returns
        -------
        None.
        """
        # kernel = np.ones((kernel_size, kernel_size), np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        self.image = cv2.erode(self.image, kernel, iterations=1)
        self.erosion = self.image.copy()

    def watershed(self, min_distance: float):
        """
        Compute the exact Euclidean distance from every binary
        pixel to the nearest zero pixel, then find peaks in this
        distance map

        Assumes that the image is already thresholded.

        Parameters
        ----------
        min_distance : float
            Minimum distance between peaks.

        Returns
        -------
        labels : ndarray
            A labeled matrix of the same type and shape as markers
            (or the input image if markers is None), where each unique
            region is assigned a unique integer label.

        markers : ndarray
            An array of markers labeled with different positive integers.
        """

        # Distance transform
        # Threshold self.blur

        # # apply self.threshold mask to self.blur
        #
        # # kernel_size = 5
        # # kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # # self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        # plt.figure()
        # plt.title('Mask')
        # plt.imshow(self.mask, cmap='gray')
        # plt.show()

        self.mask = cv2.bitwise_and(self.gray, self.gray, mask=self.thresholded)

        # plt.imshow(self.mask, cmap='gray')
        # plt.show()

        D = ndimage.distance_transform_edt(self.mask)
        self.distance_transform = D

        # plt.imshow(D)
        # plt.show()

        # Find local peaks
        localMax = peak_local_max(D, indices=False, min_distance=min_distance)
        # labels=self.closing)

        # Perform a connected component analysis on the local peaks,
        # using 8-connectivity, then apply the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        self.labels = watershed(-D, markers, mask=self.thresholded)

        return self.labels

    def apply_watershed_mask(self, label):
        """
        Create a binary mask from the watershed labels.

        Parameters
        ----------
        label : int
            Label of the mask.  

        Returns
        -------
        mask : ndarray
            A labeled matrix of the same type and shape as markers
            (or the input image if markers is None), where each unique
            region is assigned a unique integer label.
        """
        # Create a mask of the watershed labels
        if label != 0:  # label 0 is the frame
            mask = np.zeros(self.gray.shape, dtype="uint8")

            mask[self.labels == label] = 255
            return mask

    def findcontour(self, mask):
        """
        Find contours of the mask


        Parameters
        ----------
        mask : ndarray
            A labeled matrix of the same type and shape as markers
            (or the input image if markers is None), where each unique
            region is assigned a unique integer label.

        Returns
        -------
        contours : ndarray
            A list of contours.
        """

        cont, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cont

    def drawcontour(self, cont, label_id):
        """Draw the contour on the image.

        Parameters
        ----------
        cont : ndarray
            A list of contours.
        label_id : int
            The label of the contour. Can take values 1, 2, 3.
            1 == try again, 2 == pitted, 3 == pits
        Returns
        -------
        None.
        """
        if label_id == 1:
            cv2.drawContours(
                self.contour_image, cont, -1, (255, 0, 0), 2
            )  # color is blue
        elif label_id == 2:
            cv2.drawContours(
                self.contour_image, cont, -1, (0, 255, 0), 2
            )  # color is green
        else:
            cv2.drawContours(
                self.contour_image, cont, -1, (0, 0, 255), 2
            )  # color is red

    def removecroppedcontour(self, cont):
        """
        Remove the contour that is cropped by the frame

        Parameters
        ----------
        cont : ndarray
            A list of contours.

        Returns
        -------
        cont : ndarray
            A list of contours.
        """
        for cnt in cont:
            xlist = []
            ylist = []
            for ct in cnt:
                for c in ct:
                    if c[1] == 0 or c[1] == self.image.shape[0]:
                        xlist.append(c[0])
                    ylist.append(c[1])
            ymin = min(ylist)
            ymax = max(ylist)
            if ymin == 0:
                y = ymax
            else:
                y = ymin
            if len(xlist) == 0:
                return cont, False
            else:
                xmin = min(xlist)
                xmax = max(xlist)
                x = xmax - xmin
            if y < 0.3 * x:
                return cont, True
            else:
                return cont, False

    def removeerrors(self, cont):
        """
        Remove the contour that is too small or too big

        Parameters
        ----------
        cont : ndarray
            A list of contours.

        Returns
        -------
        cont : ndarray
            A list of contours.
        """
        for i in cont:
            moment = cv2.moments(i)
            # 3000 for real dataset, 1250 for synthetic dataset
            if moment["m00"] < 500 or moment["m00"] > 3500:
                return cont, True
            else:
                return cont, False

    def findcentre(self, cont):
        """
        Find the centre of the contour
        
        Parameters
        ----------
        cont : ndarray
            A list of contours.

        Returns
        ------- 
        cx : int    
            The x coordinate of the centre of the contour.
        cy : int    
            The y coordinate of the centre of the contour.    
        """
        for i in cont:
            moment = cv2.moments(i)
            if moment["m00"] != 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])
            # cv2.putText(self.contour_image, text = str(self.n), org=(cx,cy),
            #             fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0,255,0), thickness = 1, lineType=cv2.LINE_AA)
            cv2.circle(
                self.contour_image, (cx, cy), radius=0, color=(0, 0, 255), thickness=-1
            )
            # print(self.n, moment['m00'])
            return cx, cy

    def maximumdistance(self, cx, cy, cont):
        """
        Find the average distance of the contour from the centre of the image.
        
        Parameters
        ----------
        cx : int
            The x coordinate of the centre of the contour.
        cy : int
            The y coordinate of the centre of the contour.
        cont : ndarray
            A list of contours.

        Returns
        -------
        r : int
            The average distance of the contour from the centre of the image.
        """
        a = np.array((cx, cy))
        for cnt in cont:
            distancelist = []
            for ct in cnt:
                for c in ct:
                    c = np.array(c)
                    distancelist.append(np.linalg.norm(c - a))
            r = round(np.mean(distancelist))
            return r

    def sample_on_line(self, cont, num_samples=100, start_index=0, end_index=-1):
        """
        Start line at start of contour and end line at
        middle value of contour. Image is a grey scale open cv image.
        
        Parameters
        ----------
        cont : ndarray
            A list of contours.
        num_samples : int, optional
            The number of samples to be taken along the line. The default is 100.
        start_index : int, optional
            The index of the start point of the line. The default is 0.
        end_index : int, optional
            The index of the end point of the line. The default is -1.

        Returns
        -------
        values : ndarray
            The values of the pixels along the line.
        """
        cont_num_samples = len(cont[0])

        if end_index == -1:
            end_index = cont_num_samples // 2

        start = cont[0][start_index][0]
        end = cont[0][end_index][0]

        # print('[INFO]',start, end)

        # Get the line equation
        m, c = np.polyfit([start[0], end[0]], [start[1], end[1]], 1)

        # Get 100 points along the line using m and c
        x = np.linspace(start[0], end[0], num_samples)
        y = m * x + c

        """# Create contour from the line points
        line_contour = np.array([[[int(x[i]), int(y[i])]] for i in range(num_samples)])
        cv2.drawContours(self.contour_image,line_contour , -1, (0,255,0), 1) """

        # Get the values of the image along the line
        values = ndimage.map_coordinates(self.gray_scaled, np.vstack((y, x)))

        return values

    def average_sample_multiple_lines(self, cont, num_samples=100, num_lines=3):
        """
        Average multiple lines along the contour. Lines cross through
        the middle of the contour. Ensure num_lines is odd so that no duplicate
        lines are sampled.

        Parameters
        ----------
        cont : ndarray
            A list of contours.
        num_samples : int, optional
            The number of samples to be taken along the line. The default is 100.
        num_lines : int, optional
            The number of lines to be sampled. The default is 3.

        Returns
        -------
        values : ndarray
            The average values of the pixels along the lines.
        """
        values = []
        length = len(cont[0])
        # print('[INFO] Contour length:', length)

        # Create start and end indices for each line
        start_index_range = range(0, length, int(length / num_lines))
        line_counter = 0
        start_index = []
        for i in start_index_range:
            start_index.append(i)
            line_counter += 1
            if line_counter >= num_lines:
                break
        end_index = np.asarray(start_index)
        end_index = end_index + (length // 2)
        end_index = end_index % length
        end_index = end_index.tolist()

        for i in range(num_lines):
            samples = self.sample_on_line(
                cont, num_samples, start_index[i], end_index[i]
            )
            # print('[INFO] Samples:', samples)
            values.append(samples)

        values_shape = len(values)
        # print('[INFO] Values shape:', values_shape)
        return np.mean(values, axis=0)

    def find_peak_value_within_circle(self, cont, distance_from_centre, cx, cy):
        """
        Find the peak value of the contour within a circle of radius
        distance_from_centre. The circle is centred on the centre of the image.

        Parameters
        ----------
        cont : ndarray
            A list of contours.
        distance_from_centre : int
            The radius of the circle.
        cx : int
            The x coordinate of the centre of the contour.  
        cy : int
            The y coordinate of the centre of the contour.

        Returns
        -------
        max_value : int
            The maximum value of the contour within the circle.
        """
        # Create a circle mask
        circle_mask = np.zeros_like(self.gray)
        cv2.circle(circle_mask, (cx, cy), distance_from_centre, 255, -1)

        # Apply the mask to the contour
        masked_contour = cv2.bitwise_and(
            self.for_comparison, self.for_comparison, mask=circle_mask
        )

        # Get the maximum value of the contour
        max_value = np.max(masked_contour)

        return max_value

    def increase_contrast(self):
        """
        Increase the contrast of the image.
        
        Returns
        -------
        None.
        """
        lookUpTable = np.empty((1, 256), np.uint8)
        gamma = 1.5
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        self.contrast = cv2.LUT(self.for_comparison, lookUpTable)

    def apply_mask(self, mask):
        """
        Apply mask on image.
        
        Parameters
        ----------
        mask : ndarray
            A mask to be applied on the image.
            
        Returns
        -------
        masked_cherry : ndarray
            The masked image.
        """
        self.image = cv2.bitwise_and(
            self.for_comparison, self.for_comparison, mask=mask
        )
        self.masked_cherry = self.image.copy()
        return self.masked_cherry

    def crop_image_using_contour(self, cont):
        """
        Crop image using contour.

        Parameters
        ----------
        cont : ndarray
            A list of contours.

        Returns
        -------
        self.image : ndarray
            The cropped image.
        """
        # Crop the image
        x, y, w, h = cv2.boundingRect(cont[0])
        self.image = self.image[y : y + h, x : x + w]

        return self.image

    def convert_black_to_white(self, img):
        """Convert black to white.
        
        Parameters
        ----------
        img : ndarray   
            An image.

        Returns
        -------
        img : ndarray
            The image with black converted to white.
        """
        # Convert black to white
        img[img == 0] = 255
        return img

    def resize_image(self, image):
        """
        If width or height of image is smaller than 100, add white padding
        to ensure that final height and width of image is 100 x 100 pixels.

        Parameters
        ----------
        image : ndarray
            An image.

        Returns
        -------
        image : ndarray
            The resized image.
        """
        height, width = image.shape[:2]
        if height < 100:
            difference = 100 - height
            top = difference // 2
            bottom = difference - top
            left = 0
            right = 0
            image = cv2.copyMakeBorder(
                image,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )
        if width < 100:
            difference = 100 - width
            left = difference // 2
            right = difference - left
            top = 0
            bottom = 0
            image = cv2.copyMakeBorder(
                image,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )
        return image

    def find_circles(self):
        """
        Use hough circles to find circles in the image.

        Returns
        -------
        None.
        """
        # Convert to greyscale
        self.gray = cv2.cvtColor(self.for_comparison, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        self.blur = cv2.GaussianBlur(self.gray, (5, 5), 0)

        # plt.imshow(self.blur, cmap='gray')
        # plt.show()

        # Apply hough circles
        circles = cv2.HoughCircles(
            self.blur,
            cv2.HOUGH_GRADIENT,
            1,
            10,
            param1=5,
            param2=15,
            minRadius=15,
            maxRadius=30,
        )

        # Convert the (x, y) coordinates and radius of the circles to integers
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            self.circles = circles

        # Loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # Draw the circle in the output image
            cv2.circle(self.contour_image, (x, y), r, (0, 255, 0), 4)

    def invert(self):
        """
        Invert the image.

        Returns
        -------
        None.
        """
        self.inverted = cv2.bitwise_not(self.image)
        self.image = self.inverted.copy()

    def opening(self, iterations=1):
        """
        Apply opening to the image.
        
        Parameters
        ----------
        iterations : int, optional
            Number of iterations. The default is 1.
            
        Returns
        -------
        None.
        """
        kernel = np.ones((3, 3), np.uint8)
        self.open = cv2.morphologyEx(
            self.image, cv2.MORPH_OPEN, kernel, iterations=iterations
        )
        self.image = self.open.copy()


def image_processing_pipeline(image: CherryImage):
    """
    Image processing steps:
        1. Resize image
        2. Convert to greyscale
        3. Apply Gaussian blur
        4. Apply threshold
        5. Close holes
        6. Watershed

    Parameters
    ----------
    image : CherryImage
        An image.

    Returns
    -------
    None.
    """
    # Resize the image as get more circular objects
    x_stretch_factor = 1
    y_stretch_factor = 1.5
    image.resize(x_stretch_factor, y_stretch_factor)

    # Create copy of base image without any processing
    image.contour_image = image.resize.copy()
    image.for_comparison = image.resize.copy()

    # Convert to greyscale to use when evaluating models
    image.gray_scaled = cv2.cvtColor(image.resize, cv2.COLOR_BGR2GRAY)

    # Perform pyramid mean shift filtering
    spatial_radius = 21
    color_radius = 51
    image.mean_shift_filtering(spatial_radius, color_radius)

    image.gray()  # Gray the image

    # image.blur() # Gaussian blur the image

    image.threshold()  # Threshold the image
    image.closing(kernel_size=5)  # Close holes in the image

    # Perform watershed algorithm
    minimum_distance = 14
    watershed_labels = image.watershed(minimum_distance)

    return watershed_labels


def classify_object(image: CherryImage, label, gmm_pitted, gmm_pits):
    """
    For each object in the image:
        1. Apply mask to the image
        2. Find the contour of the object
        3. Remove the cropped contours
        4. Remove the objects that are too big or too small
        5. Sample using line segments
        6. Evaluate the models
        7. Draw the contour

    Parameters
    ----------
    image : CherryImage
        An image.
    label : int
        The label of the object.
    gmm_pitted : GaussianMixture
        The Gaussian Mixture Model for pitted cherries.
    gmm_pits : GaussianMixture
        The Gaussian Mixture Model for cherries with pits.
        

    Returns
    -------
    classification: str
        Can take one of the following values:
            'background'
            'cropped'
            'try_again'
            'pitted'
            'not_pitted'
    mask: np.array
    """
    

    # Binary mask of the object
    mask = image.apply_watershed_mask(label)

    if mask is None:  # This is the background
        return "background", mask

    # Find the contours of the object and skip the object if the
    # countour is cropped
    cont = image.findcontour(mask)
    finalcont, cropped = image.removecroppedcontour(cont)
    if cropped:
        return "cropped", mask

    # Check if the object is a 'try again' object
    cont, error = image.removeerrors(finalcont)
    if error:
        image.drawcontour(finalcont, label_id=1)
        return "try_again", mask

    # Sample across the objects using line segments
    sample_line = image.average_sample_multiple_lines(cont, 50, 3)

    # Save the sample line to a txt file

    # Find the centre of the object and mark it on the image
    # cx, cy = image.findcentre(finalcont)

    # Evaluate the model
    if gmm_pitted.score(sample_line.reshape(-1, 1)) > gmm_pits.score(
        sample_line.reshape(-1, 1)
    ):

        image.n += 1
        image.drawcontour(finalcont, label_id=2)
        return "pitted", mask
    else:
        image.n += 1
        image.drawcontour(finalcont, label_id=3)
        return "not_pitted", mask


def group_small_neighbours(image: CherryImage, labels):
    """
    In a 2D image with contours, some contours have been segmented into two
    parts. This function groups these two parts together by finiding the countours in the
    image. It then finds the neighbours of each contour and groups them together if they are
    close enough.

    Parameters
    ----------
    image : CherryImage
        An image.
    labels : np.array
        The labels of the objects in the image.

    Returns
    -------
    None.
    """
    # Find the contours in the image
    contours = []
    centers = []
    areas = []
    masks = []

    for label in np.unique(labels):
        # Binary mask of the object
        mask = image.apply_watershed_mask(label)

        if mask is None:  # This is the background
            continue

        # Find the contours of the object and skip the object if the
        # countour is cropped
        cont = image.findcontour(mask)
        contours.append(cont)
        print(contours[0])

        x, y = image.findcentre(cont)
        centers.append([x, y])

        areas.append(cv2.contourArea(cont))

        masks.append(mask)

    # If centers are close enough and areas below 600, group them together
    centers = np.array(centers)
    areas = np.array(areas)
    for i in range(len(centers)):
        for j in range(len(centers)):
            if i != j:
                if (
                    np.linalg.norm(centers[i] - centers[j]) < 20
                    and areas[i] < 600
                    and areas[j] < 600
                ):
                    # Combine the masks
                    masks[i] = np.bitwise_or(masks[i], masks[j])
                    # Remove the mask from the list
                    masks.pop(j)


def segment_cherries(original_image_dir, segmented_image_dir, extracted_cherries_naming, gmm_pits_dir, gmm_pitted_dir, latest_model_dir, ml_image_dir, extracted_cherries_glob):
    """
    Loads the original_image.bmp in temp folder, segments the cherries and
    saves as the segmented_image.bmp in temp folder.

    Returns
    -------
    None.
    """
    theta_r = 0.8

    # select device
    if torch.cuda.is_available():
        # Select the first available device
        device = torch.device("cuda:0")
    else:
        # Use CPU if CUDA is not available
        device = torch.device("cpu")

    # Load the image and create Image object
    image_cv2 = cv2.imread(
        original_image_dir
    )
    image = CherryImage(image_cv2)

    # ----------------------Load models ------------------------
    gmm_pits = pickle.load(
        open(gmm_pits_dir, "rb")
    )
    gmm_pitted = pickle.load(
        open(
            gmm_pitted_dir, "rb"
        )
    )

    # ----------------------Image Processing -------------------
    labels = image_processing_pipeline(image)

    # ---------------------- Group Small Neigbouring Objects ---
    # labels = group_small_neighbours(image, labels)

    contour_array = []
    rba_classification = []

    # Start RBA clock
    start = time.time()

    for obj_id, label in enumerate(np.unique(labels)):

        # ---------------------- Extract Cherries -------------------
        # Binary mask of the object
        mask = image.apply_watershed_mask(label)

        if mask is None:  # This is the background
            continue

        # Find the contours of the object and skip the object if the
        # countour is cropped
        cont = image.findcontour(mask)
        finalcont, cropped = image.removecroppedcontour(cont)
        if cropped:
            continue

        contour_array.append(finalcont)

        # Apply the mask to the image and crop
        masked_cherry = image.apply_mask(mask)

        # Find the bounding box of the cherry and crop
        x, y, w, h = cv2.boundingRect(finalcont[0])
        cropped_cherry = masked_cherry[y : y + h, x : x + w]

        # Display the cropped cherry
        # Remove the black background from cropped_cherry
        cherry = image.convert_black_to_white(cropped_cherry)

        # Ensure that width and height are 100 pixels
        cherry = image.resize_image(cherry)

        cv2.imwrite(
            extracted_cherries_naming.format(
                obj_id
            ),
            cherry,
        )

        # ----------------------Classify the object --------------
        classification, _ = classify_object(image, label, gmm_pitted, gmm_pits)
        if classification == "pitted":
            rba_classification.append(1)
        elif classification == "not_pitted":
            rba_classification.append(0)

    downsize = cv2.resize(image.contour_image, (0, 0), fx=1, fy=1 / 1.5)

    # Stop RBA clock
    end = time.time()
    print("RBA time: {}".format(end - start))

    cv2.imwrite(
        segmented_image_dir,
        downsize,
    )

    # ---------------------- Load ML model ----------------------
    # Start ML clock
    start = time.time()

    # Specify Network Architecture
    encoder = PreResNet18(2)  # 2 = number of classes
    classifier = torch.nn.Linear(encoder.fc.in_features, 2)  # 2 = number of classes
    proj_head = torch.nn.Sequential(
        torch.nn.Linear(encoder.fc.in_features, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
    )
    pred_head = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
    )
    encoder.fc = torch.nn.Identity()

    # Transfer to GPU
    encoder.to(device)
    classifier.to(device)
    proj_head.to(device)
    pred_head.to(device)

    # Load the model
    # checkpoint = load_checkpoint('C:\\Users\\Nikodem\\Desktop\\Code\\SSR_BMVC2022\cifar10\Dataset(cifar10_0.1_0.0_sym)_Model(0.9_1.0)\last1.pth.tar')
    checkpoint = load_checkpoint(
        latest_model_dir
    )
    encoder.load_state_dict(checkpoint["encoder"])
    classifier.load_state_dict(checkpoint["classifier"])
    proj_head.load_state_dict(checkpoint["proj_head"])
    pred_head.load_state_dict(checkpoint["pred_head"])

    # Set the model to evaluation mode
    encoder.eval()
    classifier.eval()

    # ---------------------- Predict the class ------------------
    # Load all images in the extracted_cherries folder

    image_paths = glob.glob(
        extracted_cherries_glob
    )
    # Sort the file names such that 10.jpg comes after 9.jpg
    image_paths.sort(key=lambda f: int(re.sub("\D", "", f)))

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    none_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Open all the images and store them in a list
    images_extracted = []
    for image_path in image_paths:
        image_extracted = Image.open(image_path)
        image_extracted = image_extracted.resize((32, 32))
        image_extracted = none_transform(image_extracted)
        images_extracted.append(image_extracted)

    # Stack the images into a tensor
    data = torch.stack(images_extracted, dim=0)
    data = data.to(device)
    feature = encoder(data)
    res = classifier(feature)
    pred = torch.argmax(res, dim=1)

    ## Replace the unconfident predictions with rba_classification
    # Apply softmax to res
    res_softmax = torch.softmax(res, dim=1)
    confidence = torch.max(res_softmax, dim=1)
    # Find the indices of the confident predictions
    unconfident_id = torch.where(confidence[0] < theta_r)[0]
    for i in unconfident_id:
        pred[i] = rba_classification[i]

    # Clean up the extracted_cherries folder
    for image_path in image_paths:
        os.remove(image_path)

    # ---------------------- Draw the contours ------------------
    # Draw the contours on the original image
    # Refresh the contour image

    image.contour_image = image.for_comparison.copy()
    # Match the labels to the predicted classes
    pred = 3 - pred
    for i in range(len(contour_array)):
        image.drawcontour(contour_array[i], label_id=pred[i])

    # Save the image
    downsize = cv2.resize(image.contour_image, (0, 0), fx=1, fy=1 / 1.5)
    cv2.imwrite(
        ml_image_dir,
        downsize,
    )

    # Stop ML clock
    end = time.time()
    print("ML time: {}".format(end - start))


if __name__ == "__main__":

    # ---------------------- Set directories ---------------------

    # Shown images
    original_image_dir = (
        "C:\\Users\\Nikodem\\Documents\\GitHub\\autonomous-training-xray\\application\\temp\\original_image.bmp"
    )
    segmented_image_dir = (
        "C:\\Users\\Nikodem\\Documents\\GitHub\\autonomous-training-xray\\application\\temp\\segmented_image.bmp"
    )
    ml_image_dir = "C:\\Users\\Nikodem\\Documents\\GitHub\\autonomous-training-xray\\application\\temp\\ml_image.bmp"

    # Extracted cherries
    extracted_cherries_naming = (
        "C:\\Users\\Nikodem\\Documents\\GitHub\\autonomous-training-xray\\application\\temp\\extracted_cherries\\{}.jpg"
    )

    extracted_cherries_glob = (
        "C:\\Users\\Nikodem\\Documents\\GitHub\\autonomous-training-xray\\application\\temp\\extracted_cherries\\*.jpg"
    )
    

    # GMM
    gmm_pits_dir = "C:\\Users\\Nikodem\\Documents\\GitHub\\autonomous-training-xray\\RBA-delphi\\gmm_pits_model.sav"
    gmm_pitted_dir = "C:\\Users\\Nikodem\\Documents\\GitHub\\autonomous-training-xray\\RBA-delphi\\gmm_pitted_model.sav"

    # Latest model
    latest_model_dir = "C:\\Users\\Nikodem\\Documents\\GitHub\\autonomous-training-xray\\application\\Win64\\Debug\\working_dir\\training_inside_delphi\\trained_model\\latest.tar"
    


    segment_cherries(original_image_dir, segmented_image_dir, extracted_cherries_naming, gmm_pits_dir, gmm_pitted_dir, latest_model_dir, ml_image_dir, extracted_cherries_glob)

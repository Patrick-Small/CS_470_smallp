##############
#Assignment 04
#CS-470
#Patrick Small
##############

###########################################################################################
#With assistance from:
#https://www.geeksforgeeks.org/create-local-binary-pattern-of-an-image-using-opencv-python/
###########################################################################################


#import stuff
import numpy as np
import cv2
from General_A04 import *

#Function that takes a (grayscale, uint8) 3x3 subimage, calculates the correct LBP label for that pixel
def getOneLBPLabel (subImage, label_type):
    
    #Local declarations
    flat = []
    LBP_binary = []
    LBP_label = 0
    switchCount = 0

    #Fill a 1 dimensional array with each value (regular order, so 1 - 9)
    for row in subImage:
        
        for pixel_value in row:
            
            flat.append (pixel_value)

    #Get the center
    center = flat[4]

    #Go clockwise from the upper left pixel and store in NEW array (1, 2, 3, 6, 9, 8, 7, 4)
    neighbors = [flat[0], flat[1], flat[2], flat[5], flat[8], flat[7], flat[6], flat[3]]

    #Use thresholding to calculate the LBP labels in binary
    for pixel_value in neighbors:
        
        if pixel_value > center:
            
            LBP_binary.append (1)
            
        else:
            
            LBP_binary.append (0)
    
    #Loop through LBP_binary
    for i in range(len(LBP_binary) - 1):
        
        #Checks if next element changes
        if LBP_binary[i] != LBP_binary[i+1]:
        
            switchCount += 1

        #Counts the number of 1's
        if LBP_binary[i] == 1:
        
            LBP_label += 1

    #Check the last element separately to avoid index out of range error
    if len(LBP_binary) > 0 and LBP_binary[-1] == 1:
        
        LBP_label += 1

    #If it's not uniform, then it's labeled as 9
    if switchCount > 2:
        
        LBP_label = 9

    """ OLD CODE [This calculates full LBP, need to calculate uniform]
    #So it goes from lowest to highest value
    LBP_binary.reverse () 
    
    #You have a list of binary values, but you should turn it into a decimal value (256 possible labels)
    for i in range (len(LBP_binary)):
        
        LBP_label += LBP_binary[i] * (2 ** i)
    """
    
    #Return the decimal label
    return LBP_label


#Function that takes a (grayscale, uint8) image then generates a uniform, rotation-invariant LBP label image
def getLBPImage (image, label_type):
    
    #Local declarations
    row, col = image.shape
    LBP_image = np.zeros ((row, col), dtype = np.uint8) #This is the eventual output image
    
    #Create a padded image (using copyMakeBorder from assignment)
    paddedImage = cv2.copyMakeBorder (image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = 0)
    
    #Loop through each pixel in the image
    for r in range (1, row + 1):
        
        for c in range (1, col + 1):
            
            #Cut out appropriate subImage
            subImage = paddedImage [r - 1 : r + 2, c - 1 : c + 2]
            
            #Call getOneLBPLabel to get the correct output label per pixel
            outputLabel = getOneLBPLabel (subImage, label_type)
            
            #Store in LBP_image
            LBP_image [r - 1, c - 1] = outputLabel
    
    
    #Return the uniform, rotation-invariant LBP label image
    return LBP_image


#Function that takes a LBP label image then computes the uniform LBP histogram
def getOneRegionLBPFeatures (subImage, label_type):
    
    #Flatten the image
    flatImage = subImage.flatten ()
    
    #Get histogram (Make sure histogram length is 10 elements.) (bin_edges doesn't get used)
    hist, bin_edges = np.histogram (flatImage, bins = 10, range = (0, 10), density = True)
    
    #Normalize the histogram by dividing it by the total number of pixels
    histNormal = hist / np.sum (hist)
    
    #Return the correct LBP histogram
    return histNormal


#Function that creates a list of histograms and returns them all
def getLBPFeatures (featureImage, regionSideCnt, label_type):
    
    #Local declarations
    row, col = featureImage.shape
    allHists = []
    
    #Compute the subregion width and height in pixels
    subWidth = col // regionSideCnt
    subHeight = row // regionSideCnt
    
    #Loop through each possible subregion, row by row then col by col
    for r in range (regionSideCnt):
        
        for c in range (regionSideCnt):
            
            #Get the starting point
            rowStart = r * subHeight
            colStart = c * subWidth
            
            #Extract the subImage (subregions do NOT overlap)
            subImage = featureImage [rowStart : rowStart + subHeight, colStart : colStart + subWidth]
            
            #Call getOneRegionLBPFeatures() to get the one subImage's histogram
            subImageHist = getOneRegionLBPFeatures (subImage, label_type)
            
            #Add it to the list
            allHists.append (subImageHist)
    
    #Convert your list of histograms to an np.array()
    allHists = np.array (allHists)
    
    #Then reshape so that it is a flat array
    allHists = np.reshape (allHists, (allHists.shape[0] * allHists.shape[1],))
    
    #Return allHists
    return allHists
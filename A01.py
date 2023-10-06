##############
#Assignment 01
#CS-470
#Patrick Small
##############

#import stuff
import sys
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import gradio as gr

#Creates a numpy array and an unnormlaized histogram. Returns the histogram.
def create_unnormalized_hist(image):

    #Assume image is grayscale, it's a shape (height, width) and is uint8 dtype
    #Create a numpy array of type "float32" and shape (256,)
    #Code below is not needed
    #hist = np.array((256,), dtype = "float32")

    #Make and return the UNNORMALIZED histogram
    return cv2.calcHist([image], [0], None, [256], [0,256])[:,0]
    
#Recieves an unnormalized histogram, and returns it normalized (using np.sum)
def normalize_hist(hist):

    #Normalize the histogram by dividing hist by the sum of all elements
    hist = hist/np.sum(hist)
    
    #Return it!
    return hist

#Recieves a normalized histogram, and computes (& returns) the CDF
def create_cdf(nhist):
    
    #Create a numpy array first
    temp = np.array((256,), dtype = "float32")
    
    #Using numpy np.cumsum to calculate the CDF
    cdf = np.cumsum(nhist)
    
    #Return CDF
    return cdf

def get_hist_equalize_transform(image, do_stretching, do_cl=False, cl_thresh=0):
    
    #Calculate the unnormalized histogram
    unNormHist= create_unnormalized_hist(image)
    
    #Normalize it
    normHist = normalize_hist(unNormHist)
    
    #Make the CDF
    cdf = create_cdf(normHist)
    
    #Stretch the histogram (maybe lol)
    if(do_stretching == True):
        
        #Perform Histogram stretching
        #This makes it so the first element is 0
        cdf = cdf - cdf[0]
        
        #rescale so the last element is still 1
        cdf = cdf/cdf[255]
        
    
    #Create transformation function
    #Changes all values BETWEEN first and last
    cdf = cdf * 255
    
    #Taken straight from assignment
    int_transform = cv2.convertScaleAbs(cdf) [:,0]
    
    #Return int_transform
    return int_transform

def do_histogram_equalize(image, do_stretching):
    
    #Copy your image to output
    output = np.copy(image)
    
    #Get your transformation function
    t = get_hist_equalize_transform(output, do_stretching)
    
    #For each pixel, get the value, use transformation to get new value, store in output
    for i in range (0, image.shape[0]):
        for j in range (0, image.shape[1]):
            store = image[i,j]
            output[i,j] = t[store]
            
    #return output
    return output

#Next three blocks of code are copied & pasted in.
#Used in order to run program with Gradio
def intensity_callback(input_img, do_stretching):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    output_img = do_histogram_equalize(input_img, do_stretching)
    return output_img

def main():
    demo = gr.Interface(fn=intensity_callback,
                        inputs=["image", "checkbox"],
                        outputs=["image"])
    demo.launch()
    
if __name__ == "__main__":
    main()

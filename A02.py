##############
#Assignment 01
#CS-470
#Patrick Small
##############

#######################################################################
#With assistance from:
#https://www.w3schools.com/python/python_file_open.asp
#https://www.w3schools.com/python/ref_string_split.asp
#https://www.digitalocean.com/community/tutorials/numpy-zeros-in-python
#######################################################################


#import stuff
#import sys
import numpy as np
#import tensorflow as tf
import cv2
#import matplotlib.pyplot as plt
import gradio as gr


#Allowed OpenCV functionality
########################
#Loading/saving an image
#Window display
#Grayscale conversion
#convertScaleAbs
#flip
#copyMakeBorder
########################

#Reads a file, does some *stuff*, and returns a kernel
def read_kernel_file(filepath):
    
    #open a file for reading
    openFile = open(filepath)
    
    #Read the first line (needs to be readline(), not read())
    line = openFile.readline()
    
    #Split the line into tokens by space (string split method)
    tokens = line.split(" ")
    
    #Grab and convert 1st & 2nd tokens into ints
    rowCnt = int(tokens[0])
    colCnt = int(tokens[1])
    
    #Create a np array of zeros of shape (rowCnt, colCnt)
    kernel = np.zeros((rowCnt, colCnt))
    
    #This makes the token value start at 2
    val = 2
    
    #Loop through and store correct token (converted to float)
    for r in range(rowCnt):
        
        for c in range(colCnt):
           
            #Store the value in the kernel
            kernel[r][c] = float(tokens[val])
            
            #Increase the value
            val = val + 1
    
    #First 7 tests to load the filter worked, yeehaw
    
    #Return the kernel
    return kernel


#Creates a padded image, executes correlation, then returns an ouput image
def apply_filter(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    
    #Cast image & kernel to float64
    image = image.astype("float64")
    kernel = kernel.astype("float64")
    
    #Rotate kernel 180 degrees to perform convolution
    kernel = cv2.flip (kernel, -1)
    
    #Create padded image
    #Get the amount of padding as the h & w of the kernel int /2
    padH = kernel.shape[0]//2
    padW = kernel.shape[1]//2
    
    #Use copyMakeBorder to make the padded image
    padImage = cv2.copyMakeBorder(image, padH, padH, padW, padW, borderType = cv2.BORDER_CONSTANT, value = 0)
    
    #Create a floating-point np array (size of original image)
    output = np.zeros(image.shape)
    
    #For each possible center pixel (r, c), do correlation
    for r in range(image.shape[0]):
        
        for c in range(image.shape[1]):
           
           #Grab the subimage from the padded image
           subImage = padImage[r: (r + kernel.shape[0]), c: (c + kernel.shape[1])]
           
           #Multiply the subimage by the kernel
           filtervals = subImage * kernel
           
           #Get the sum of these values
           value = np.sum(filtervals)
           
           #Write this to your output image
           output[r, c] = value
           
    #if convert_uint8 = T, use convertScaleAbs using alpha & beta
    if (convert_uint8 == True):
        
        output = cv2.convertScaleAbs(output, alpha = alpha, beta = beta)
    
    #return output image
    return output


#Copied directly from assignment page (everyone & 470 only)
def filtering_callback(input_img, filter_file, alpha_val, beta_val):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    kernel = read_kernel_file(filter_file.name)
    output_img = apply_filter(input_img, kernel, alpha_val, beta_val)
    return output_img

def main():
    demo = gr.Interface(fn=filtering_callback,
                        inputs=["image",
                                "file",
                                gr.Number(value=0.125),
                                gr.Number(value=127)],
                        outputs=["image"])
    demo.launch() 

# Later, at the bottom ##I'm assuming that means below the *def main()* function
if __name__ == "__main__": 
    main()
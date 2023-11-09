##############
#Assignment 03
#CS-470
#Patrick Small
##############

########################################################################################
#With assistance from:
#https://stackoverflow.com/questions/61689930/all-centers-coincide-on-k-means-clustering
########################################################################################


#import stuff
import numpy as np
import cv2
from skimage.segmentation import slic

#Going with the method used in the assignment
#"TFA" means "Taken From Assignment"

def find_WBC(image):
    
    #Declarations
    boundingBoxes = []
    
    #Get superpixel groups
    #n_segments = 25 and compactness = 25 gave the best accuracy I think?
    #compactness = 50 brought both acc & IOU up
    #compactness = 75 seemed to be the most ideal
    segments = slic(image, n_segments = 25, compactness = 75, start_label = 0)
    
    #cnt is how many groups there are [TFA]
    cnt = len(np.unique(segments))
    
    #Make the numpy array to hold the mean color per superpixel [TFA]
    group_means = np.zeros((cnt, 3), dtype = "float32")
    
    #Loop through each superpixel index [TFA]
    for specific_group in range(cnt):
       
        #Create mask image [TFA]
        mask_image = np.where(segments == specific_group, 225, 0).astype("uint8")
       
        #Add channel dimension back in to the mask [TFA sorta]
        mask_image = np.expand_dims(mask_image, axis = -1)
       
        #Compute mean value per group and slice the result [TFA]
        group_means [specific_group] = cv2.mean(image, mask = mask_image) [0:3]
        
    #Use K-means on group mean colors (into 4 color groups) [Stack Overflow assistance]
    #Switching up the 10 and 1 made no noticable differences in accuracy and IOU, hmmmmm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    ret, bestLabels, centers = cv2.kmeans (group_means, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    #Find the K-means group with mean closest to blue [Stack Overflow assistance]
    blue = np.array([255, 0, 0])
    closest = np.argmin(np.linalg.norm(centers - blue, axis = 1))
    
    #Set that k-means group to white and the rest to black [TFA sorta]
    centers = np.array([np.array([0, 0, 0])] * 4)
    centers[closest] = np.array([255, 255, 255])
    
    #Determind the new colors for each superpixel group [TFA]
    centers = np.uint8 (centers)
    colors_per_clumps = centers [bestLabels.flatten()]
    
    #Recolor the superpixels with their new group colors [TFA]
    cell_mask = colors_per_clumps[segments]
    
    #Convert the recolored stuff into a 1936 tv
    cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
    
    #get disjoint blobs from cell_mask [Slide 14 of region slide deck]
    retval, labels = cv2.connectedComponents(cell_mask)
    
    #For each blob group (except for 0 cus its the background) [TFA sorta]
    for i in range(1, retval):
        
        #Get the coordinates of the pixels that belong to that group
        coords = np.where(labels == i)
        
        #As long as the coordinates are found -
        if coords[0].any():
            
            # - get the min and max (x & y) coordinates to get the bounding box
            ymin, xmin = np.min(coords[0]), np.min(coords[1])
            ymax, xmax = np.max(coords[0]), np.max(coords[1])
            
            #Add the bounding box to the list
            boundingBoxes.append((ymin, xmin, ymax, xmax))
            
    #Exit     
    return boundingBoxes
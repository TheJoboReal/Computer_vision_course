import numpy as np
import cv2
from matplotlib import pyplot as plt



def cov(img, img_annotated):
    assert img is not None, "Failed to load image."
    
    pixels = np.reshape(img, (-1, 3))           # Reshape the image to a 2D array

    mask = cv2.inRange(img_annotated, (253, 253, 253), (255, 255, 255))           # Create a mask for the annotated pixels
    mask_pixels = np.reshape(mask, (-1))                                    # Reshape the mask to a 1D array
    cv2.imwrite('../output/ex10_annotation_mask.jpg', mask)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)

    # Determine mean value, standard deviations and covariance matrix
    # for the annotated pixels.
    # Using cv2 to calculate mean and standard deviations
    mean, std = cv2.meanStdDev(img, mask = mask)                            # Calculate mean and standard deviation of the annotated pixels

    pixels = np.reshape(img, (-1, 3))                       # Reshape the image to a 2D array
    mask_pixels = np.reshape(mask, (-1))                    # Reshape the mask to a 1D array
    annot_pix_values = pixels[mask_pixels == 255, ]         # Extract the color values of the annotated pixels
    avg = np.average(annot_pix_values, axis=0)              # Calculate the average color values of the annotated pixels
    cov = np.cov(annot_pix_values.transpose())              # Calculate the covariance matrix of the color values of the annotated pixels

    print("Mean color values of the annotated pixels")
    print(avg)
    print("Covariance matrix of color values of the annotated pixels")
    print(cov)

    return avg, cov


def mahalanobis(img, img_annotated):
  assert img is not None, "Failed to load image."
  
  reference_color, covariance_matrix = cov(img, img_annotated)

  pixels = np.reshape(img, (-1, 3))                                         # Reshape the image to a 2D array

  # Calculate the euclidean distance to the reference_color annotated color.
  shape = pixels.shape                                                # Get the shape of the image                    
  diff = pixels - np.repeat([reference_color], shape[0], axis=0)      # Calculate the difference between the pixels and the reference color
  inv_cov = np.linalg.inv(covariance_matrix)                          # Calculate the inverse of the covariance matrix
  moddotproduct = diff * (diff @ inv_cov)                             # Calculate the modified dot product
  mahalanobis_dist = np.sum(moddotproduct,                            # Calculate the Mahalanobis distance
      axis=1)                                                         # Calculate the Mahalanobis distance
  mahalanobis_distance_image = np.reshape(
      mahalanobis_dist, 
        (img.shape[0],
         img.shape[1]))                                               # Reshape the Mahalanobis distance to the original image shape

  # Scale the distance image and export it.
  mahalanobis_distance_image = 0.005 * 255 * mahalanobis_distance_image / np.max(mahalanobis_distance_image)
  cv2.imshow('mahalanobis_distance_image', mahalanobis_distance_image)
  cv2.waitKey(0)
  return mahalanobis_distance_image


# def close_objects(img):
#   kernel = np.ones((50, 50), np.uint8)  # Define a kernel for morphological operations
#   closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # Perform morphological closing operation on the image
#   cv2.imshow('Closed Objects', closed_img)
#   cv2.waitKey(0)
#   return closed_img

def close_objects(img, kernel_dim1, kernel_dim2,iteration):
   kernel = np.ones((kernel_dim1, kernel_dim2), np.uint8)  # Define a kernel for morphological operations
   eroded_img = cv2.erode(img, kernel, iterations=iteration)  # Perform morphological closing operation on the image
   opened_img = cv2.dilate(eroded_img, kernel, iterations=iteration)  # Perform morphological closing operation on the image

   closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel)  # Perform morphological closing operation on the image
   return closed_img
  
  

def count_objects(mahalanobis_img):
   ret, threshhold = cv2.threshold(mahalanobis_img, 0.1, 255, cv2.THRESH_BINARY)    # Apply a threshold to the Mahalanobis distance image
   cv2.imshow('threshhold', 255 * threshhold)
   print(mahalanobis_img)
   print(threshhold)
   cv2.waitKey(0)

   threshhold = close_objects(threshhold, 22, 22, 3)
   cv2.imshow('threshhold', 255 * threshhold)
   print(mahalanobis_img)
   print(threshhold)
   cv2.waitKey(0)

   contours, hierarchy = cv2.findContours(threshhold.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # Find the contours in the thresholded image
   print("Number of objects: ", len(contours)) # Print the number of objects in the image

   # Draw a circle above the center of each of the detected contours.
   for contour in contours:
        M = cv2.moments(contour)
        if(cv2.contourArea(contour) > 0):
          cx = int(M['m10'] / M['m00'])
          cy = int(M['m01'] / M['m00'])
          cv2.circle(threshhold, (cx, cy), 40, (0, 0, 255), 2)

   cv2.imshow('Detected Contours', threshhold)
   cv2.waitKey(0)

   return len(contours)


img = cv2.imread("images/Cropped.jpg")
img_annotated = cv2.imread("images/Cropped_annotate.jpg")

maha = mahalanobis(img, img_annotated)

count_objects(maha)
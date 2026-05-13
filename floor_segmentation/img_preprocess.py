# a set of image preprocessing utilities for floor segmentation
import numpy as np
import cv2
import skimage

## Some images may have black padding, we should try to extract the actual region of the image
def extract_colored_region(img):
  # add dim for greyscale images
  if len(img.shape) == 2:
    img = np.expand_dims(img, axis=2)

  colored_areas = np.any(img > 0, axis=2) # checks the color channels to see if any are colored, returning a (H, W) array of booleans
  rows_w_color = np.any(colored_areas, axis=1) # checks rows if it has color, returning array of size (H,) of booleans
  cols_w_color = np.any(colored_areas, axis=0) # checks columns if it has color, return (W,)
  # print(cols_with_color) # array where columns with color have TRUE. False otherwise.

  # get bounding area
  row_indices = np.where(rows_w_color)[0]  # gets all indices with color (row)
  col_indices = np.where(cols_w_color)[0]
  min_row = row_indices[0]
  max_row = row_indices[-1]
  min_col = col_indices[0]
  max_col = col_indices[-1]

  actual_region = img[min_row:max_row+1, min_col:max_col+1] # inclusive slicing
  return actual_region

def get_sobel_edge_map(img_path):
  img = cv2.imread(img_path)
  img = extract_colored_region(img) # crop out black borders if they exist
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  s_channel = hsv_img[:, :, 1] # get saturation channel
  gx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=3)
  gy = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1, ksize=3)

  edge_map = np.sqrt(gx**2 + gy**2)
  edge_map = (edge_map / np.max(edge_map) * 255).astype(np.uint8) # normalize to [0, 255]
  return edge_map

def get_superpixel_img(img_path):
  img = cv2.imread(img_path) 
  img = extract_colored_region(img) # crop out black borders if they exist
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  # SLIC superpixels
  superpixel_img = skimage.segmentation.slic(hsv_img, n_segments=100, compactness=10)
  return superpixel_img

def get_area_info(superpixel_labels):
  area_stats = dict()
  H, W = superpixel_labels.shape
  total_labels = np.max(superpixel_labels)
  for i in range(total_labels):
    pixels_in_label = np.sum(superpixel_labels[superpixel_labels == i + 1] / (i + 1))
    area_stats[i + 1] = pixels_in_label

  # pixels in all labels sum up to the image size (or gets very close to it)
  assert np.sum(np.array(list(area_stats.values()))) == H * W
  return area_stats
import numpy as np
import torch
from pyproj import Transformer, CRS
from typing import List, Optional
from tqdm import tqdm

FINAL_H = 17
FINAL_W = 18

# resize array
def resize_array(input_array: np.array, new_shape: tuple, verbose: bool = False, mode="hybrid") -> np.array:
      # Get the current shape of the input array
      current_shape = input_array.shape
      if verbose:
        print("current shape: ", current_shape)
        print("new shape: ", new_shape)
      # Calculate the padding for each dimension
      pad_height = np.max( np.array([ 0, new_shape[0] - current_shape[0]]) )
      pad_width = np.max(  np.array([0, new_shape[1] - current_shape[1]]) )

      # Case where the new_shape is strictily smaller than the original shape
      # Only case we manually handled, cv2 is used otherwise
      if pad_width == 0 and pad_width == 0:
        # Define how many element of the original matrix are aggregated along the dimensions
        # So define the dimensions of sub blocks to aggregate
        if mode == "hybrid":
          sub_height = round(current_shape[0]/new_shape[0])
          sub_width = round(current_shape[1]/new_shape[1])
        elif mode == "original":
          sub_height = current_shape[0]//new_shape[0]
          sub_width = current_shape[1]//new_shape[1]

        remaining_heigth = current_shape[0]%new_shape[0]
        remaining_width = current_shape[1]%new_shape[1]

        if verbose:
          print( sub_height , sub_width )
          print( remaining_heigth , remaining_width )

        out_array = np.zeros(new_shape)

        for i in range(new_shape[0]):
          for j in range(new_shape[1]):
            # inside box
            if ( i != new_shape[0] - 1 ) and (j != new_shape[1]-1):
              sub_block = input_array[ i*sub_height : (i+1)*sub_height , j*sub_width : (j+1)*sub_width ]
            # last_width
            elif ( i != new_shape[0] - 1 ) and (j == new_shape[1]-1):
              sub_block = input_array[ i*sub_height : (i+1)*sub_height , j*sub_width :  ]
            # last_height
            elif ( i == new_shape[0] - 1 ) and (j != new_shape[1]-1):
              sub_block = input_array[ i*sub_height : , j*sub_width : (j+1)*sub_width ]
            # both last
            elif ( i == new_shape[0] - 1 ) and (j == new_shape[1]-1):
              sub_block = input_array[ i*sub_height : , j*sub_width : ]
            if verbose:
              print( sub_block, "\nWith mean= ", np.mean(sub_block) )
            out_array[i,j] = np.mean(sub_block)
      return out_array

# covert text matrix
def convert_matrix(matrix, verbose=False):
  # find number of different labels
  max_value_of_matrix = np.max(matrix)
  # build a matrix with all zeros of the final shape (n_lables, ...original_shape)
  in_shape = matrix.shape
  new_matrix = np.zeros((max_value_of_matrix, in_shape[0], in_shape[1]))

  # fill each channel, that correspond to a label, with 1 where there was that label
  for i in range(max_value_of_matrix):
    new_matrix[i, :, :] = np.where(matrix == i+1, 1, 0)

  if verbose:
    print(matrix, "\n", new_matrix)
  return new_matrix


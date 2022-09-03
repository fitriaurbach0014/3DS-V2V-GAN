import os
import numpy as np
import tensorflow as tf
import nibabel as nib
from tensorflow.keras.utils import to_categorical
from nilearn.image import resample_img
import skimage.transform as skTrans
#import sklearn.utils.class_weight as class_weight
import matplotlib.pyplot as plt
from augmentation import *

folder_path = '/content/drive/MyDrive/Fitria/vox2vox_modif/'

def rescale_affine(input_affine, voxel_dims=[1, 1, 1], target_center_coords= None):

    # Initialize target_affine
    target_affine = input_affine.copy()
    # Decompose the image affine to allow scaling
    u,s,v = np.linalg.svd(target_affine[:3,:3],full_matrices=False)
    
    # Rescale the image to the appropriate voxel dimensions
    s = voxel_dims
    
    # Reconstruct the affine
    target_affine[:3,:3] = u @ np.diag(s) @ v

    # Set the translation component of the affine computed from the input
    # image affine if coordinates are specified by the user.
    if target_center_coords is not None:
        target_affine[:3,3] = target_center_coords
    return target_affine

def load_img(img_files):
    ''' Load one image and its target form file
    '''
    N = len(img_files)
    y = nib.load(img_files[N-1])

    #Find Target Affine
    target_affine = y.affine.copy()
    # Calculate the translation part of the affine
    spatial_dimensions = (y.header['dim'] * y.header['pixdim'])[1:4]
    # Calculate the translation affine as a proportion of the real world
    # spatial dimensions
    image_center_as_prop = y.affine[0:3,3] / spatial_dimensions    
    # Calculate the equivalent center coordinates in the target image
    dimensions_of_target_image = (np.array([1,1,1]) * np.array([256,256,128]))
    target_center_coords =  dimensions_of_target_image * image_center_as_prop 
    voxel_dims = [1,1,1]
    target_affine = rescale_affine(target_affine,voxel_dims,target_center_coords)

    y = resample_img(y, target_affine,target_shape=(256,256,128), interpolation='linear')
    y = y.get_fdata(dtype='float32', caching='unchanged')
    
    X_norm = np.empty((256,256,128,1))
    for channel in range(N-1):
      X = nib.load(img_files[channel])

      #Find Target Affine
      target_affine_img = X.affine.copy()
    
      # Calculate the translation part of the affine
      spatial_dimensions_img =(X.header['dim'] * X.header['pixdim'])[1:4]
    
      # Calculate the translation affine as a proportion of the real world
      # spatial dimensions
      image_center_as_prop_img = X.affine[0:3,3] / spatial_dimensions
    
      # Calculate the equivalent center coordinates in the target image
      dimensions_of_target_image_img = (np.array([1,1,1]) * np.array([256,256,128]))
      target_center_coords_img =  dimensions_of_target_image_img * image_center_as_prop_img 

      voxel_dims = [1,1,1]
      target_affine_img = rescale_affine(target_affine_img,voxel_dims,target_center_coords_img)

      #Resample image
      X = resample_img(X, target_affine=target_affine_img,target_shape=(256,256,128), interpolation='linear')
      X = X.get_fdata(dtype='float32', caching='unchanged')
      
      brain = X[X!=0] 
      brain_norm = np.zeros_like(X) # background at -100
      norm = (brain - np.mean(brain))/np.std(brain)
      brain_norm[X!=0] = norm
      X_norm[:,:,:,channel] = brain_norm        
        
    X_norm = X_norm[:,:,:,:]    
    del(X, brain, brain_norm)
    
    return X_norm, y
    
def visualize(X):
    """
    Visualize the image middle slices for each axis
    """
    a,b,c = X.shape
    
    plt.figure(figsize=(15,15))
    plt.subplot(131)
    plt.imshow(np.rot90(X[a//2, :, :]), cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(np.rot90(X[:, b//2, :]), cmap='gray')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(X[:, :, c//2], cmap='gray')
    plt.axis('off')

#280, 320, 128   
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=4, dim=(256,256,128), n_channels=1, n_classes=2, shuffle=True, augmentation=False, patch_size= 128, n_patches=1):
        'Initialization'
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data     
        X, y = self.__data_generation(list_IDs_temp)
        if self.augmentation == True:
            X, y = self.__data_augmentation(X, y)
        
        if index == self.__len__()-1:
            self.on_epoch_end()
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
  
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, IDs in enumerate(list_IDs_temp):
            # Store sample
            X[i], y[i] = load_img(IDs)
            
        if self.augmentation == True:
            return X.astype('float32'), y
        else:
            return X.astype('float32'), to_categorical(y, self.n_classes)

    def __data_augmentation(self, X, y):
        'Apply augmentation'
        X_aug, y_aug = patch_extraction(X, y, sizePatches=self.patch_size, Npatches=self.n_patches)
        X_aug, y_aug = aug_batch(X_aug, y_aug)     
        return X_aug, to_categorical(y_aug, self.n_classes)

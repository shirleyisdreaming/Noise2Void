
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
import zipfile


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

filenumber=2

datagen = N2V_DataGenerator()
imgs = datagen.load_imgs_from_directory(directory = "datalshr/")
for count in range(filenumber):
 patch_shape = (96,96)
 X = datagen.generate_patches_from_list(imgs[count:count+1], shape=patch_shape)
 if(count!=(filenumber-1)):
  X_val = datagen.generate_patches_from_list(imgs[count+1:count+2], shape=patch_shape)
 else:
  X_val = datagen.generate_patches_from_list(imgs[count-1:count], shape=patch_shape)

 config = N2VConfig(X, unet_kern_size=3,train_steps_per_epoch=int(X.shape[0]/128), train_epochs=100, train_loss='mse', batch_norm=True,train_batch_size=128, n2v_perc_pix=0.198, n2v_patch_shape=(64, 64),n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5)
     
 #config = N2VConfig(X, unet_kern_size=3,train_steps_per_epoch=int(X.shape[0]/128), train_epochs=1, train_loss='mse', batch_norm=True,
                   #train_batch_size=128, n2v_perc_pix=0.198, n2v_patch_shape=(64, 64),
                   #n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5)
                   
 model_name = 'n2v_2D'
# the base directory in which our model will live
 basedir = 'modelsforfig/model%d'%(100565+count)
# We are now creating our network model.
 model = N2V(config, model_name, basedir=basedir)

 history = model.train(X, X_val)
 train_loss=history.history['loss']
 print(train_loss)
 n2v_mse=history.history['n2v_mse']
 print(n2v_mse)
 val_n2v_mse=history.history['val_n2v_mse']
 print(val_n2v_mse)
 xc=range(100)
 plt.figure()
 plt.plot(xc,n2v_mse)
 plt.savefig('n2v_mse.png')
 plt.figure()
 plt.plot(xc,val_n2v_mse)
 plt.savefig('val_n2v_mse.png')

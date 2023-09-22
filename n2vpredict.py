from n2v.models import N2V
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread
from csbdeep.io import save_tiff_imagej_compatible

filenumber=2
for count in range(filenumber):
 model_name = 'n2v_2D'
 basedir = 'models/model%d'%(100565+count)
 model = N2V(config=None, name=model_name, basedir=basedir)

 input_train = imread('datalshr/shade_LSHR_voi_nf_median_%d.tif'%(100565+count))

 pred_train = model.predict(input_train, axes='YX', n_tiles=(2,1))


 save_tiff_imagej_compatible('n2vpredict/shade_LSHR_voi_nf_n2v_%d.tif'%(100565+count), pred_train, axes='YX')


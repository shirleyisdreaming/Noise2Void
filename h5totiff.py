import h5py, imageio
import numpy as np 

filename="/home/beams12/S1IDUSER/xiaoxu/n2v/sharma_dmi_sam5_nf_mr_3pt0s/sharma_sam5_mr.h5"

with h5py.File(filename, "r") as f:
 print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
 a_group_key = list(f.keys())[0]
 ds_arr = f[a_group_key][()]  # returns as a numpy array
    
for g1 in range(ds_arr.shape[0]):
 median=ds_arr[g1]
 g2=g1+248516
 imageio.imwrite(f"/home/beams12/S1IDUSER/xiaoxu/n2v/sharma_dmi_sam5_nf_mr_3pt0s/sharma_dmi_sam5_nf_0pt8s_{g2}.tif",median.astype('uint16'))
            
    
    
 
 
 
 
  

import h5py, glob, skimage.io
import numpy as np 

def main():
    imgs = []
    for fn in sorted(glob.glob('/home/beams12/S1IDUSER/xiaoxu/n2v/sharma_dmi_sam5_nf_3pt0s/*.tif'))[:]:
        _img = skimage.io.imread(fn)[None]
        imgs.append(_img)
    imgs  = np.concatenate(imgs, axis=0)


    imgs = imgs.astype(np.int32)
    fp = h5py.File('/home/beams12/S1IDUSER/xiaoxu/n2v/sharma_dmi_sam5_nf_3pt0s/sharma_sam5_mr.h5', 'w')
    for g in range(1):
        s = g * 181
        e = (g+1) * 181
        median = np.median(imgs[s:e], axis=0)
        imgs[s:e] = imgs[s:e] - median
        imgs[s:e][imgs[s:e] < 0] = 0
        #skimage.io.imsave(f"median-exp{g+1}.tiff", median)
        fp.create_dataset(f'median-exp{g+1}',data=median)
        


    fp.create_dataset('image', data=imgs, dtype=np.int32)
    fp.close()

if __name__ == "__main__":
    main()

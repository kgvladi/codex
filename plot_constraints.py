#!/usr/bin/env python
# coding: utf-8
#--Plotting the constraints in (x,y) basis
#--howto launch: python3.10 plot_constraints.py 

from pandas import *
import numpy as np
import sys
from __future__ import print_function
import pickle
import tifffile as tf
import xml.etree.ElementTree as et

from matplotlib import pyplot as plt
import matplotlib

import deepcell
from deepcell.utils.plot_utils import create_rgb_image,make_outline_overlay
from deepcell_toolbox.processing import histogram_normalization
from deepcell.utils.transform_utils import inner_distance_transform_2d
from deepcell.applications import Mesmer

#--reading the original CODEX file 
tif=tf.TiffFile('FF_AITL_3524.qptiff')
page=tif.pages[0]
series=tif.series[0]
biomarker_list=[]
for page in tif.series[0].pages:
  tmp_marker=et.fromstring(page.description).find('Biomarker').text
  biomarker_list.append(tmp_marker)

#--selecting initial (x,y) region of the tissue
n1=4600
n2=5200
n3=17000
n4=18000
numpx=n2-n1
numpy=n4-n3

#--extracting the nucleus 
nuclear_imgf=tif.series[0].pages[biomarker_list.index('DAPI')].asarray()
nuclear_img=nuclear_imgf[n1:n2,n3:n4]
#--extracting the membrane
mem_imgf=tif.series[0].pages[biomarker_list.index('CD45RO')].asarray()
mem_img=mem_imgf[n1:n2,n3:n4]

codex_img=np.stack([nuclear_img,mem_img],axis=2)
codex_img=np.expand_dims(codex_img,axis=0)
codex_img=histogram_normalization(codex_img)

#--cell segmentation
app=Mesmer()
segmentation_predictions_nuc=app.predict(codex_img,image_mpp=0.5,compartment='nuclear')

#reading the file with the constraints in (x,y) basis
dataf = read_pickle('tumor6_segcell_max_constraints_grid_corr.pkl')
nc=len(dataf.columns)
#plotting
matplotlib.rcParams['font.size']=22.0
fig,axs=plt.subplots(3,3,sharex='all',sharey='all',figsize=(20,20))
k=0
l=0
idx=0
datam=dataf.to_numpy()
for i in range(9):
 vals=np.zeros((numpx,numpy))
 r=0
 for p in range(numpx):
    for q in range(numpy):
        vals[p,q]=datam[r,i]
#        vals[p,q]=np.float64(dataf.values[r,i+9])
#        vals[p,q]=np.float64(dataf.values[r,i+18])
#        vals[p,q]=np.float64(dataf.values[r,i+27])
        if (vals[p,q]==20):
            vals[p,q]=np.nan
        r=r+1
 maxv=np.max(vals)
 minv=np.min(vals)
 title='G'+str(i)
# title='G'+str(i+9)
# title='G'+str(i+18)
# title='G'+str(i+27)
 codex_img=np.stack([nuclear_img,vals],axis=2)
 codex_img=np.expand_dims(codex_img,axis=0)
 overlay_data=make_outline_overlay(rgb_data=codex_img,predictions=segmentation_predictions_nuc)
 if (i==0):
   pcm=axs[k,l].imshow(overlay_data[0,0:numpx,0:numpy,1],interpolation="none",vmin=-0.03,vmax=0,cmap='Spectral_r')
 else:
   pcm=axs[k,l].imshow(overlay_data[0,0:numpx,0:numpy,1],interpolation="none",vmin=-0.05,vmax=0.05,cmap='bwr')    
 fig.colorbar(pcm,ax=axs[k,l],fraction=0.046,pad=0.04)
 axs[k,l].set_title(title)
# axs[0,0].text(10,20,data.index[i]+" data",color='white',fontsize=30.0)
# axs[0,0].text(10,190,"Max = "+'%.2f'%norm,color='white',fontsize=30.0)
 l=l+1
 if (l%3==0): 
   l=0
   k=k+1
filew='Constraints_tumor6_segcell_max_corr_9.pdf'
plt.tight_layout()
plt.savefig(filew)#,bbox_inches='tight')
plt.close(fig)

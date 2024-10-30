#!/usr/bin/env python
# coding: utf-8
#--Plotting the fits by different number of constraints for all biomarkers
#--howto launch: python3.10 plot.py 

from pandas import *
import numpy as np
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

data = read_excel('tumor6_segcell_max_lambdas.xlsx',header=0,index_col=0)
numm=len(data.index)
print(data.index)
nc=len(data.columns)
for i in range(numm):
 marker=data.index[i]
 file=marker+'_tumor6_segmaxx.pkl'
 dataf = read_pickle(file)
#original data
 vals=dataf.to_numpy()
 norm=np.max(vals)
 vals=vals/norm
#plotting
 matplotlib.rcParams['font.size']=22.0
 fig,axs=plt.subplots(4,3,sharex='all',sharey='all',figsize=(20,20))
 k=0
 l=0
 codex_img=np.stack([nuclear_img,vals],axis=2)
 codex_img=np.expand_dims(codex_img,axis=0)
 overlay_data=make_outline_overlay(rgb_data=codex_img,predictions=segmentation_predictions_nuc)
 pcm=axs[k,l].imshow(overlay_data[0,0:num,0:num,1],interpolation="none",vmin=0,vmax=1,cmap='Spectral_r')    
 fig.colorbar(pcm,ax=axs[k,l],fraction=0.046,pad=0.04)
 axs[k,l].text(30,100,"Data",color='white',fontsize=25.0)
 title=data.index[i]
 axs[k,l].set_title(title)
#constraints fit 
 k=0
 l=1
 for j in range(11):
   filer='tumor6_segcell_max_corr_'+str(j)+'_fit_'+data.index[i]+'.pkl'
   fit = read_pickle(filer)
   fitm=fit.to_numpy()
   vals=np.reshape(fitm,(numpx,numpy),order='F')
   norm=np.max(vals)
   vals=np.transpose(vals)
   vals=vals/norm
   codex_img=np.stack([nuclear_img,vals],axis=2)
   codex_img=np.expand_dims(codex_img,axis=0)
   overlay_data=make_outline_overlay(rgb_data=codex_img,predictions=segmentation_predictions_nuc)
   pcm=axs[k,l].imshow(overlay_data[0,0:numpx,0:numpy,1],interpolation="none",vmin=0,vmax=1,cmap='Spectral_r')    
   fig.colorbar(pcm,ax=axs[k,l],fraction=0.046,pad=0.04)
   axs[k,l].text(30,100,str(j),color='white',fontsize=25.0)
   l=l+1
   if (l%3==0): 
     l=0
     k=k+1
    
 filew=data.index[i]+'_fit_tumor6_segcell_max_corr.pdf'
 plt.tight_layout()
 plt.savefig(filew)#,bbox_inches='tight')
 plt.close(fig)

#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from pandas import *
import numpy as np
import sys,re

import tifffile as tf
import xml.etree.ElementTree as et

from skimage.transform import resize
from matplotlib import pyplot as plt
import matplotlib
import deepcell
from deepcell.utils.plot_utils import create_rgb_image,make_outline_overlay
from deepcell_toolbox.processing import histogram_normalization
from deepcell.utils.transform_utils import inner_distance_transform_2d

import pickle
from deepcell.applications import Mesmer


#################################
#regexps
stn=re.compile('\d+')
dd=re.compile('[-]?\d+[.]\d+')
fnc=re.compile('\w+[.]\w+')
fn=re.compile('\w+')
#####################################################
tif=tf.TiffFile('FF_AITL_3524.qptiff')
page=tif.pages[0]
print(page.shape)
series=tif.series[0]
print(series.shape)
print(et.fromstring(page.description).find('Biomarker').text)
biomarker_list=[]
for page in tif.series[0].pages:
  tmp_marker=et.fromstring(page.description).find('Biomarker').text
  biomarker_list.append(tmp_marker)
print(len(biomarker_list))
#print(biomarker_list)
umm=len(biomarker_list)
nuclear_imgf=tif.series[0].pages[biomarker_list.index('DAPI')].asarray()
num=1000
#nuclear_img=nuclear_imgf[4400:5400,14500:15500]
#nuclear_img=nuclear_imgf[5200:6200,13700:14700]
#nuclear_img=nuclear_imgf[6000:7000,16000:17000]
#nuclear_img=nuclear_imgf[6000:7000,20000:21000]
#R5
#nuclear_img=nuclear_imgf[26500:27500,21700:22700]
#R6
nuclear_img=nuclear_imgf[27000:28000,20600:21600]

mem_imgf=tif.series[0].pages[biomarker_list.index('CD45RO')].asarray()
#mem_img=mem_imgf[4400:5400,14500:15500]
#mem_img=mem_imgf[5200:6200,13700:14700]
#mem_img=mem_imgf[6000:7000,16000:17000]
#mem_img=mem_imgf[6000:7000,20000:21000]
#mem_img=mem_imgf[26500:27500,21700:22700]
mem_img=mem_imgf[27000:28000,20600:21600]

codex_img=np.stack([nuclear_img,mem_img],axis=2)
codex_img=np.expand_dims(codex_img,axis=0)
codex_img=histogram_normalization(codex_img)

app=Mesmer()
segmentation_predictions_nuc=app.predict(codex_img,image_mpp=0.5,compartment='nuclear')
#################################
matplotlib.rcParams['font.size']=22.0
dataf = read_pickle('tumor6_segcell_max_constraints_grid_corr.pkl')

nc=len(dataf.columns)
ng=num
dng=ng*ng
fig,axs=plt.subplots(3,3,sharex='all',sharey='all',figsize=(20,20))
k=0
l=0
idx=0
datam=dataf.to_numpy()
for i in range(9):
 vals=np.zeros((ng,ng))
 r=0
 for p in range(ng):
    for q in range(ng):
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
   pcm=axs[k,l].imshow(overlay_data[0,0:num,0:num,1],interpolation="none",vmin=-0.03,vmax=0,cmap='Spectral_r')
 else:
   pcm=axs[k,l].imshow(overlay_data[0,0:num,0:num,1],interpolation="none",vmin=-0.05,vmax=0.05,cmap='bwr')    
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

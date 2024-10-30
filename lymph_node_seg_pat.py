#!/usr/bin/env python
# coding: utf-8
#################
#--Script functionalities: creating the approximate image of the data 
#by taking only the max value of the response within each cell 
#  -> reads the original data file
#  -> takes a small region in (x,y) coordinates and performs cell segmentation
#  -> replaces the value of the response within each cell with maximal value of the response in each cell
#--howto launch: python3.10 lymph_node_seg_pat.py $marker 
##################

from pandas import *
import numpy as np
from __future__ import print_function
import sys
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

#marker for which the data are plotted
marker=str(sys.argv[1])
print(marker)

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
#extracting the marker response
marker_imgf=tif.series[0].pages[biomarker_list.index(marker)].asarray()
marker_img=marker_imgf[n1:n2,n3:n4]

codex_img=np.stack([nuclear_img,mem_img],axis=2)
codex_img=np.expand_dims(codex_img,axis=0)
codex_img=histogram_normalization(codex_img)

#--cell segmentation
app=Mesmer()
segmentation_predictions_nuc=app.predict(codex_img,image_mpp=0.5,compartment='nuclear')
seg=DataFrame(segmentation_predictions_nuc[0,0:nump,0:nump,0])

numcells=np.max(seg)

datacells=np.zeros((numpx,numpy,numcells))
dataf=np.zeros((numpx,numpy))
for i in range(numpx):
 for j in range(numpy):
   if (seg.values[i,j] != 0):       
    for k in range(numcells):
     if (seg.values[i,j] == k ):
        datacells[i,j,k]=marker_img[i,j]
for k in range(numcells):
 maxcells=np.max(datacells[:,:,k])
# print(k,maxcells)
 for i in range(numpx):
  for j in range(numpy):
    if (seg.values[i,j] == k):
      dataf[i,j]=maxcells
data=DataFrame(dataf)
#file=marker+'_tumor6_segmaxx.xlsx'
#data.to_excel(file)
file=marker+'_tumor6_segmaxx.pkl'
data.to_pickle(file)

###plotting
marker_data=DataFrame(marker_img)
k=0
l=0
codex_img=np.stack([marker_data,dataf],axis=2)
codex_img=np.expand_dims(codex_img,axis=0)
overlay_data=make_outline_overlay(rgb_data=codex_img,predictions=segmentation_predictions_nuc)
fig,axs=plt.subplots(1,2,sharex='all',sharey='all',figsize=(20,20))
pcm=axs[0].imshow(overlay_data[0,0:numpx,0:numpy,0],interpolation="none",cmap='Spectral_r')    
fig.colorbar(pcm,ax=axs[0],fraction=0.046,pad=0.04)
pcm=axs[1].imshow(overlay_data[0,0:numpx,0:numpy,1],interpolation="none",cmap='Spectral_r')    
fig.colorbar(pcm,ax=axs[1],fraction=0.046,pad=0.04)
title=marker
axs[0].set_title(title)
filew=marker+'_segmaxx_tumor6.pdf'
plt.tight_layout()
plt.savefig(filew)#,bbox_inches='tight')
plt.close(fig)

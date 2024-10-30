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

from deepcell.applications import Mesmer

import pickle

#import maxfuse as mf
#import seaborn as sns
#import anndata as ad
#import scanpy as sc
#import errno
#################################
#regexps
stn=re.compile('\d+')
dd=re.compile('[-]?\d+[.]\d+')
fnc=re.compile('\w+[.]\w+')
fn=re.compile('\w+')
#####################################################
#
#
marker=str(sys.argv[1])
print(marker)
#marker='Collagen IV'
tif=tf.TiffFile('FF_AITL_3524.qptiff')
page=tif.pages[0]
series=tif.series[0]
biomarker_list=[]
for page in tif.series[0].pages:
  tmp_marker=et.fromstring(page.description).find('Biomarker').text
  biomarker_list.append(tmp_marker)

nump=1000
nuclear_imgf=tif.series[0].pages[biomarker_list.index('DAPI')].asarray()
#nuclear_img=nuclear_imgf[4400:5400,14500:15500]
#nuclear_img=nuclear_imgf[5200:6200,13700:14700]
#nuclear_img=nuclear_imgf[6000:7000,16000:17000]
#nuclear_img=nuclear_imgf[6000:7000,20000:21000]
#nuclear_img=nuclear_imgf[26500:27500,21700:22700]
nuclear_img=nuclear_imgf[27000:28000,20600:21600]

mem_imgf=tif.series[0].pages[biomarker_list.index('CD45RO')].asarray()
#mem_img=mem_imgf[4400:5400,14500:15500]
#mem_img=mem_imgf[5200:6200,13700:14700]
#mem_img=mem_imgf[6000:7000,16000:17000]
#mem_img=mem_imgf[6000:7000,20000:21000]
#mem_img=mem_imgf[26500:27500,21700:22700]
mem_img=mem_imgf[27000:28000,20600:21600]

marker_imgf=tif.series[0].pages[biomarker_list.index(marker)].asarray()
#marker_img=marker_imgf[4400:5400,14500:15500]
#marker_img=marker_imgf[5200:6200,13700:14700]
#marker_img=marker_imgf[6000:7000,16000:17000]
#marker_img=marker_imgf[6000:7000,20000:21000]
#marker_img=marker_imgf[26500:27500,21700:22700]
marker_img=marker_imgf[27000:28000,20600:21600]

codex_img=np.stack([nuclear_img,mem_img],axis=2)
codex_img=np.expand_dims(codex_img,axis=0)
codex_img=histogram_normalization(codex_img)

app=Mesmer()
segmentation_predictions_nuc=app.predict(codex_img,image_mpp=0.5,compartment='nuclear')
seg=DataFrame(segmentation_predictions_nuc[0,0:nump,0:nump,0])
#print(seg.shape)
#file='Segmentation_mask.xlsx'
#seg.to_excel(file)
numcells=np.max(seg)

datacells=np.zeros((nump,nump,numcells))
dataf=np.zeros((nump,nump))
for i in range(nump):
 for j in range(nump):
   if (seg.values[i,j] != 0):       
    for k in range(numcells):
     if (seg.values[i,j] == k ):
        datacells[i,j,k]=marker_img[i,j]
for k in range(numcells):
 maxcells=np.max(datacells[:,:,k])
# print(k,maxcells)
 for i in range(nump):
  for j in range(nump):
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
pcm=axs[0].imshow(overlay_data[0,0:nump,0:nump,0],interpolation="none",cmap='Spectral_r')    
fig.colorbar(pcm,ax=axs[0],fraction=0.046,pad=0.04)
pcm=axs[1].imshow(overlay_data[0,0:nump,0:nump,1],interpolation="none",cmap='Spectral_r')    
fig.colorbar(pcm,ax=axs[1],fraction=0.046,pad=0.04)
title=marker
axs[0].set_title(title)
filew=marker+'_segmaxx_tumor6.pdf'
plt.tight_layout()
plt.savefig(filew)#,bbox_inches='tight')
plt.close(fig)

#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from pandas import *
import numpy as np
import sys,re

import tifffile as tf
import pickle as pic
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
numm=len(biomarker_list)
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
data = read_excel('tumor6_segcell_max_lambdas.xlsx',header=0,index_col=0)
numm=len(data.index)
print(data.index)
nc=len(data.columns)
ng=num
dng=ng*ng
for i in range(numm):
 marker=data.index[i]
 file=marker+'_tumor6_segmaxx.pkl'
 dataf = read_pickle(file)
 fig,axs=plt.subplots(4,3,sharex='all',sharey='all',figsize=(20,20))
#original data
 vals=dataf.to_numpy()
 norm=np.max(vals)
 vals=vals/norm
#plotting
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
   vals=np.reshape(fitm,(ng,ng),order='F')
   norm=np.max(vals)
   vals=np.transpose(vals)
   vals=vals/norm
   codex_img=np.stack([nuclear_img,vals],axis=2)
   codex_img=np.expand_dims(codex_img,axis=0)
   overlay_data=make_outline_overlay(rgb_data=codex_img,predictions=segmentation_predictions_nuc)
   pcm=axs[k,l].imshow(overlay_data[0,0:num,0:num,1],interpolation="none",vmin=0,vmax=1,cmap='Spectral_r')    
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

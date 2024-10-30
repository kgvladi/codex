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
nuclear_img=nuclear_imgf[6000:7000,20000:21000]

mem_imgf=tif.series[0].pages[biomarker_list.index('CD45RO')].asarray()
#mem_img=mem_imgf[4400:5400,14500:15500]
#mem_img=mem_imgf[5200:6200,13700:14700]
#mem_img=mem_imgf[6000:7000,16000:17000]
mem_img=mem_imgf[6000:7000,20000:21000]

codex_img=np.stack([nuclear_img,mem_img],axis=2)
codex_img=np.expand_dims(codex_img,axis=0)
codex_img=histogram_normalization(codex_img)

app=Mesmer()
segmentation_predictions_nuc=app.predict(codex_img,image_mpp=0.5,compartment='nuclear')
#################################
matplotlib.rcParams['font.size']=22.0
data = read_excel('tumor4_segcell_max_lambdas.xlsx',header=0,index_col=0)
dataf = read_excel('tumor4_segcell_max_data_nz.xlsx',header=0,index_col=0)
numm=len(data.index)
print(len(data.index),len(dataf.columns))
print(data.index)
print(dataf.columns)
nc=len(data.columns)
ng=num
dng=ng*ng
marker='CD45RO'
fig,ax=plt.subplots()
#original data
filer='tumor4_segcell_max_specific_fit_CD45RO.xlsx'
fit = read_excel(filer,header=0,index_col=0)
fitm=fit.to_numpy()
vals=np.reshape(fitm,(ng,ng),order='F')
norm=np.max(vals)
vals=vals/norm
#plotting
codex_img=np.stack([nuclear_img,vals],axis=2)
codex_img=np.expand_dims(codex_img,axis=0)
overlay_data=make_outline_overlay(rgb_data=codex_img,predictions=segmentation_predictions_nuc)
pcm=ax.imshow(overlay_data[0,0:num,0:num,1],interpolation="none",vmin=0.0,vmax=1,cmap='Spectral_r')    
#fig.colorbar(pcm,ax=ax,ifraction=0.046,pad=0.04)
title='CD45RO'
ax.set_title(title)
filew='Specificfit_CD45RO_tumor4_segcell_max.pdf'
plt.tight_layout()
plt.savefig(filew)#,bbox_inches='tight')
plt.close(fig)

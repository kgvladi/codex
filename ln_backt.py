#!/usr/bin/env python
# coding: utf-8
####################
#--Script transforms the constraints in cell basis to the constraints in (x,y) basis 
#accounting for outliers
#--how to launch: python3.10 ln_backt.py
####################

from pandas import *
import numpy as np
import tifffile as tf
import xml.etree.ElementTree as et
import pickle
from __future__ import print_function

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
dnp=numpx*numpy
#number of the contraints
numm=26

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
seg=DataFrame(segmentation_predictions_nuc[0,0:numpx,0:numpy,0])
numcells=np.max(seg)

#--reading the file with the constraints in the cell basis
file_gs="tumor6_segcell_max_constraints_L.xlsx"
gs = read_excel(file_gs,header=0,index_col=0)
gsm=gs.to_numpy()
print(gsm.shape)

#--remove outliers in the constraints
for i in range(numm):
    meanv=np.mean(gsm[:,i])
    stdv=np.std(gsm[:,i])
    thrp=meanv+3*stdv
    thrm=meanv-3*stdv
    print(i,meanv,stdv,thrp,thrm)
    for j in range(numcells):
      if (gsm[j,i] > thrp or gsm[j,i] < thrm):
        gsm[j,i]=20
        
#--foreach constraint    
#--map cell into xy position
gs_grid=np.zeros((dnp,numm))
l=0
for i in range(numpx):
  for j in range(numpy):
   if (seg.at[i,j] != 0):       
    for k in range(numcells):
     if (seg.at[i,j] == k+1 ):
      for q in range(numm):
        gs_grid[l,q]=gsm[k,q]
   else:
     #--outliers
      for q in range(numm):
        gs_grid[l,q]=20
   l=l+1
data=DataFrame(gs_grid,columns=gs.columns)
file='tumor6_segcell_max_constraints_grid_corr.pkl'
data.to_pickle(file)

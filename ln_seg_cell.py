#!/usr/bin/env python
# coding: utf-8

from pandas import *
import numpy as np
import tifffile as tf
import xml.etree.ElementTree as et
import pickle
from matplotlib import pyplot as plt
import matplotlib
import sys

import deepcell
from deepcell.utils.plot_utils import create_rgb_image,make_outline_overlay
from deepcell_toolbox.processing import histogram_normalization
from deepcell.utils.transform_utils import inner_distance_transform_2d
from deepcell.applications import Mesmer

marker=str(sys.argv[1])
tif=tf.TiffFile('FF_AITL_3524.qptiff')
page=tif.pages[0]
series=tif.series[0]
biomarker_list=[]
for page in tif.series[0].pages:
  tmp_marker=et.fromstring(page.description).find('Biomarker').text
  biomarker_list.append(tmp_marker)

n1=4600
n2=5200
n3=17000
n4=18000
numpx=n2-n1
numpy=n4-n3

nuclear_imgf=tif.series[0].pages[biomarker_list.index('DAPI')].asarray()
nuclear_img=nuclear_imgf[n1:n2,n3:n4]
mem_imgf=tif.series[0].pages[biomarker_list.index('CD45RO')].asarray()
mem_img=mem_imgf[n1:n2,n3:n4]

marker_imgf=tif.series[0].pages[biomarker_list.index(marker)].asarray()
marker_img=marker_imgf[n1:n2,n3:n4]

codex_img=np.stack([nuclear_img,mem_img],axis=2)
codex_img=np.expand_dims(codex_img,axis=0)
codex_img=histogram_normalization(codex_img)

app=Mesmer()
segmentation_predictions_nuc=app.predict(codex_img,image_mpp=0.5,compartment='nuclear')
seg=DataFrame(segmentation_predictions_nuc[0,0:numpx,0:numpy,0])
numcells=np.max(seg)

index_cells=[]
dataf=np.zeros((numcells,1))
for k in range(numcells):
 datacells=np.zeros((numpx,numpy))
 for i in range(numpx):
  for j in range(numpy):
   if (seg.at[i,j] != 0):       
     if (seg.at[i,j] == k ):
        datacells[i,j]=marker_img[i,j]
 dataf[k,0]=np.max(datacells)
 index_cells.append(str(k))
data=DataFrame(dataf,index_cells,columns=[marker])
file=marker+'_tumor23_segcell_max.xlsx'
data.to_excel(file)

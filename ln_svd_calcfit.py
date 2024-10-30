#!/usr/bin/env python
# coding: utf-8

#import matplotlib.pyplot as plt
from __future__ import print_function
from pandas import *
import numpy as np
import sys,re


markers=[]
markers.append('DAPI')
#markers.append('EpCAM')
markers.append('ICOS')
markers.append('Ki67')
markers.append('CD4')
markers.append('CD8')
markers.append('Podoplanin')
markers.append('CD21')
markers.append('CD11c')
markers.append('CD3e')
markers.append('p21')
markers.append('IDO1')
markers.append('FOXP3')
markers.append('CXCL13')
markers.append('HLA-DR')
markers.append('EBNA')
markers.append('TOX')
markers.append('yH2AX') 
markers.append('MPO')
markers.append('LAG3')
markers.append('HMBG1')
markers.append('CD163')
markers.append('CD68')
markers.append('CD45RO') 
markers.append('CD141')
markers.append('CD14')
markers.append('CD44')

numm=len(markers)
nump=1000
dnp=nump*nump
file2="tumor6_segcell_max_constraints_grid_corr.pkl"
gs=read_pickle(file2)
file4="tumor6_segcell_max_lambdas.xlsx"
lams=read_excel(file4,header=0,index_col=0)
#filt=np.zeros((numm,numm))
#for i in range(numm):
# for j in range(numm):
#     if (np.abs(lams.values[i,j]) >= 0.05*np.abs(lams.values[i,0])):
#       filt[i,j]=lams.values[i,j]
#     else:
#       filt[i,j]=np.nan
#fileres='tumor6_segcell_sum_lambdas_filtered.xlsx'
#res=DataFrame(filt,lams.index,lams.columns)
#res.to_excel(fileres) 

for sp in range(numm):
   sprotein=markers[sp]
   print(sprotein)
   slam=lams.loc[sprotein]
   vals=np.zeros((1,numm))
   for i in range(numm):
     vals[0,i]=slam.values[i]
   slam_new=DataFrame(vals,index=['lambdas'],columns=gs.columns)
   lamg=concat([slam_new,gs])
   lamg.sort_values(by='lambdas',axis=1,inplace=True,ascending=False,key=abs)
   lamf=lamg.to_numpy()
   #accuracy
   #computation by 2 constraints
   #exp(-lambdas*G)
   err=np.zeros((numm,dnp))
   for j in range(dnp):
     k1=0
     for qr in range(12):
      if (lamf[j+1,qr] != 20):
          k1=k1+lamf[0,qr]*lamf[j+1,qr]
          datafit=np.exp(-k1)
          err[qr,j]=datafit
      else:
        err[qr,j]=0.0

   for qr in range (12):
     file1='tumor6_segcell_max_corr_'+str(qr)+'_fit_'+sprotein+'.pkl'
     res1=DataFrame(err[qr,:],gs.index,columns=[sprotein])
     res1.to_pickle(file1) 

 



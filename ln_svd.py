#!/usr/bin/env python
# coding: utf-8

#import matplotlib.pyplot as plt
from __future__ import print_function
from pandas import *
import numpy as np
import sys,re


#################################
#regexps
stn=re.compile('\d+')
dd=re.compile('[-]?\d+[.]\d+')
fnc=re.compile('\w+[.]\w+')
fn=re.compile('\w+')
#################################################
#functions
def A_times_Atranspose(df):
    transpos = np.transpose(df)
    A_times_Atranspose = np.dot(df, transpos)
    return A_times_Atranspose
def At_times_A(df):
    transpos = np.transpose(df)
    At_times_A = np.dot(transpos,df)
    return At_times_A
#################################################

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
file='ICOS_tumor23_segcell_max.xlsx'
datam = read_excel(file,header=0,index_col=0)
datam=datam.drop(index=0)
numcells=len(datam.index)
data=np.zeros((numcells,numm))
l=0
check=0
for i in range(numm): 
    name=markers[i]
    file=name+'_tumor23_segcell_max.xlsx'
    datam = read_excel(file,header=0,index_col=0)
    datam=datam.drop(index=0)
    for m in range(numcells):
     data[m,i]=datam.values[m,0]
push=DataFrame(data,index=datam.index,columns=markers)
file='tumor23_segcell_max_data.xlsx'
push.to_excel(file)     
############################################
#count the number of zeroes
dnp=numcells
nzeros=np.zeros((numm,1))
for i in range(numm):
  for j in range(dnp):
     if (np.float64(push.values[j,i]) == 0):
       nzeros[i]=nzeros[i]+1
nz=DataFrame(nzeros, push.columns)  
outputfile="tumor23_segcell_max_nzeros.xlsx"
nz.to_excel(outputfile)
#replace zeros with a random number
rng=np.random.default_rng()
for i in range(numm):
  for j in range(dnp):
     if (np.float64(push.values[j,i]) == 0):
       k=20
       push.values[j,i]=np.exp(-k)
       if (np.float64(push.values[j,i]) == 0):
         k=20
         push.values[j,i]=np.exp(-k)
         
sur=np.zeros((dnp,numm))
#go to surprisal
for i in range(dnp):
 for j in range(numm):
   k=np.float64(push.values[i,j])
   sur[i,j]=-np.log(k)
#do svd
dot_p = At_times_A(sur)
eigenvaluesp, eigenvectorsp = np.linalg.eigh(dot_p, UPLO="L")
eigenvectorsg=np.matmul(sur,eigenvectorsp)
eigenvaluesp=np.abs(eigenvaluesp)
eigenvaluesp=np.sqrt(eigenvaluesp)
for i in range(numm):
   for j in range(dnp):
      eigenvectorsg[j,i]=eigenvectorsg[j,i]/eigenvaluesp[i]

for i in range(numm):
 for j in range(numm):
   k=eigenvaluesp[j]
   eigenvectorsp[i,j]=eigenvectorsp[i,j]*k

#print("Eigenvalues (L)")
#file1=inputfile[:-4]+"c_eigenvalues_L.xlsx"
file2="tumor23_segcell_max_constraints_L.xlsx"
file3="tumor23_segcell_max_eigenvalues_p.xlsx"
file4="tumor23_segcell_max_lambdas.xlsx"
#
##res1=DataFrame(eigenvaluesg[::-1],dtype=float)  
##res1.to_excel(file1) 
res2=DataFrame(eigenvectorsg, push.index,dtype=float)  
res2=res2.iloc[:,::-1]
res2.to_excel(file2) 
res3=DataFrame(eigenvaluesp[::-1],dtype=float)  
res3.to_excel(file3) 
res4=DataFrame(eigenvectorsp, push.columns,dtype=float)  
res4=res4.iloc[:,::-1]
res4.to_excel(file4) 
#=======================================================
##lams=read_excel(file4,header=0,index_col=0)
#lams=DataFrame(eigenvectorsp, index=push.columns,dtype=float)  
##gs=read_excel(file2,header=0,index_col=0)
#gs=DataFrame(eigenvectorsg, index=push.index,dtype=float)  
#for sp in range(numm):
#   sprotein=markers[sp]
#   print(sprotein)
#   slam=lams.loc[sprotein]
#   vals=np.zeros((1,numm))
#   for i in range(numm):
#     vals[0,i]=slam.values[i]
#   slam_new=DataFrame(vals,index=['lambdas'],columns=gs.columns)
#   lamg=concat([slam_new,gs])
#   lamg.sort_values(by='lambdas',axis=1,inplace=True,ascending=False,key=abs)
##   file='tumor4_segcell_joined_'+sprotein+".xlsx"
# #  lamg.to_excel(file) 
#   
#   #accuracy
#   #computation by 2 constraints
#   #exp(-lambdas*G)
#   err=np.zeros((numm,dnp))
#   #for i in range(np):
#   for j in range(dnp):
#     k1=0
#     for qr in range(numm):
#      k1=k1+lamg.values[0,qr]*lamg.values[j+1,qr]
#      datafit=np.exp(-k1)
#      err[qr,j]=datafit
#   for qr in range (numm):
##   qr=numm-1
#    file1='tumor4_segcell_'+str(qr)+'_fit_'+sprotein+'.xlsx'
#    res1=DataFrame(err[qr,:],gs.index,columns=[sprotein])
#    res1.to_excel(file1) 

 



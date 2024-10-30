#!/usr/bin/env python
# coding: utf-8
##############
#Script performs svd for the dataset: computes lambdas (characteristic vectors) and constraints 
##############

from pandas import *
import numpy as np

#functions for svd 
def A_times_Atranspose(df):
    transpos = np.transpose(df)
    A_times_Atranspose = np.dot(df, transpos)
    return A_times_Atranspose
def At_times_A(df):
    transpos = np.transpose(df)
    At_times_A = np.dot(transpos,df)
    return At_times_A
#################################################
#biomarkers list in a specific order
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
#getting dimensions
#number of biomarkers
numm=len(markers)
file='ICOS_tumor23_segcell_max.xlsx'
datam = read_excel(file,header=0,index_col=0)
datam=datam.drop(index=0)
#number of cells
numcells=len(datam.index)
#reading prepared data files for each marker and stacking columns
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
nzeros=np.zeros((numm,1))
for i in range(numm):
  for j in range(numcells):
     if (np.float64(push.values[j,i]) == 0):
       nzeros[i]=nzeros[i]+1
nz=DataFrame(nzeros, push.columns)  
outputfile="tumor23_segcell_max_nzeros.xlsx"
nz.to_excel(outputfile)
#replace zeros with a random number
rng=np.random.default_rng()
for i in range(numm):
  for j in range(numcells):
     if (np.float64(push.values[j,i]) == 0):
       k=20
       push.values[j,i]=np.exp(-k)
       if (np.float64(push.values[j,i]) == 0):
         k=20
         push.values[j,i]=np.exp(-k)
         
sur=np.zeros((numcells,numm))
#go to surprisal
for i in range(numcells):
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
   for j in range(numcells):
      eigenvectorsg[j,i]=eigenvectorsg[j,i]/eigenvaluesp[i]

for i in range(numm):
 for j in range(numm):
   k=eigenvaluesp[j]
   eigenvectorsp[i,j]=eigenvectorsp[i,j]*k

file2="tumor23_segcell_max_constraints_L.xlsx"
file3="tumor23_segcell_max_eigenvalues_p.xlsx"
file4="tumor23_segcell_max_lambdas.xlsx"

res2=DataFrame(eigenvectorsg, push.index,dtype=float)  
res2=res2.iloc[:,::-1]
res2.to_excel(file2) 
res3=DataFrame(eigenvaluesp[::-1],dtype=float)  
res3.to_excel(file3) 
res4=DataFrame(eigenvectorsp, push.columns,dtype=float)  
res4=res4.iloc[:,::-1]
res4.to_excel(file4) 

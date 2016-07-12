import numpy as np
import time
import os
import pyfftw
import time
import scipy
from evtk.hl import gridToVTK
import sys
from Classes import gridclass, FFTclass, variables
from padding import *
from RHSfunctions import *
from pylab import *

## Check if variables exist
#==============================================
if 'cfl' in globals():			     #|
  pass					     #|
else:				             #|
  cfl = -dt				     #|
if 'fft_type' in globals():	  	     #|
  pass					     #|
else:				             #|
  fft_type = 'pyfftw'	  		     #|

#==============================================

# Make Solution Directory if it does not exist
if (mpi_rank == 0):
  if not os.path.exists('3DSolution'):
     os.makedirs('3DSolution')

# Initialize Classes. 
#=====================================================================
myFFT = FFTclass(N1,N2,N3,nthreads,fft_type,Npx,Npy,num_processes,comm)
grid = gridclass(N1,N2,N3,x,y,z,kc,num_processes,L1,L3,mpi_rank)
main = variables(grid,u,v,w,t,dt,et,nu,myFFT,Re_tau)

ucheck = myFFT.myifft3D( myFFT.myfft3D(main.u))
checkVal = np.linalg.norm(main.u - ucheck)
print('FFT CHECK = ' + str(checkVal) )
if ( checkVal >= 1e-1 ):
  sys.stdout.write('ERROR! FFT Check routines have error -> ifft(fft(u)) = ' + str(checkVal) + ' \n')
  sys.stdout.write('Quitting! \n')
  sys.stdout.flush()
  exit(0)


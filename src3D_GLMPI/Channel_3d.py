import numpy as np
import time
import os
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
myFFT = FFTclass(N1,N2,N3,nthreads,fft_type,Npx,Npy,num_processes,comm,mpi_rank)
grid = gridclass(N1,N2,N3,x,y,z,kc,num_processes,L1,L3,mpi_rank,comm)
main = variables(grid,u,v,w,t,dt,et,nu,myFFT,Re_tau)
#====================================================================

main.iteration = 0
main.save_freq = save_freq

### Main time integration loop
t0 = time.time()

ucheck = myFFT.myifft3D( myFFT.myfft3D(main.u))
checkVal = np.linalg.norm(main.u - ucheck)
#if (mpi_rank == 0):
#  print('FFT CHECK = ' + str(checkVal) )
#  if ( checkVal >= 1e-1 ):
#    sys.stdout.write('ERROR! FFT Check routines have error -> ifft(fft(u)) = ' + str(checkVal) + ' \n')
#    sys.stdout.write('Quitting! \n')
#    sys.stdout.flush()
#    exit(0)


#========================================================================
while (main.t < main.et):
  #------------- Save Output ------------------
  if (main.iteration%main.save_freq == 0):
    div = checkDivergence(main,grid)
    main.u = myFFT.myifft3D(main.uhat)
    main.v = myFFT.myifft3D(main.vhat)
    main.w = myFFT.myifft3D(main.what)
    divG = allGather_spectral(div,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npx)
    uGlobal = allGather_physical(main.u,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npy)
    vGlobal = allGather_physical(main.v,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npy)
    wGlobal = allGather_physical(main.w,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npy)
    if (mpi_rank == 0):
      string = '3DSolution/PVsol' + str(main.iteration)
      string2 = '3DSolution/npsol' + str(main.iteration)
      np.savez(string2,u=uGlobal,v=vGlobal,w=wGlobal)
      #main.p = myFFT.myifft3D(main.phat)
      sys.stdout.write("===================================================================================== \n")
      sys.stdout.write('t = '  + str(main.t) + '   Wall time = ' + str(time.time() - t0) + '\n' )
      sys.stdout.write('Div = ' + str( np.linalg.norm(divG) )  + '\n')
  
      sys.stdout.flush()
      
      gridToVTK(string, grid.xG,grid.yG,grid.zG, pointData = {"u" : np.real(uGlobal.transpose()) , \
        "v" : np.real(vGlobal.transpose()) , "w" : np.real(wGlobal.transpose()) } ) #, \
       #363       "p" : np.real(pdummy.transpose())} )

  #---------------------------------------------
  # advance by Adams Bashforth/Crank Nicolson
  main.iteration += 1
  advance_AdamsCrank(main,grid,myFFT)
  main.t += main.dt
#========================================================================



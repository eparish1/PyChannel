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

## Check if variables exist
#==============================================
if 'turb_model' in globals():                #|
  pass                                       #|
else:                                        #|
  turb_model = 'DNS'                         #|
if 'cfl' in globals():			     #|
  pass					     #|
else:				             #|
  cfl = -dt				     #|
if 'fft_type' in globals():	  	     #|
  pass					     #|
else:				             #|
  fft_type = 'pyfftw'	  		     #|
if 'tau0' in globals():                      #|
  pass                                       #|
else:                                        #|
  tau0 = 0.1                                 #|
if 'Cs' in globals():                        #|
  pass                                       #|
else:                                        #|
  Cs = 0.16                                  #|
if 'iteration_start' in globals():           #|
  pass                                       #|
else:                                        #|
  iteration_start = 0                        #|
if 'computeStats' in globals():              #|
  pass                                       #|
else:                                        #|
  computeStats = 1                           #|
#==============================================

# Make Solution Directory if it does not exist
if (mpi_rank == 0):
  if not os.path.exists('3DSolution'):
     os.makedirs('3DSolution')

# Initialize Classes. 
#=====================================================================
myFFT = FFTclass(N1,N2,N3,nthreads,fft_type,Npx,Npy,num_processes,comm,mpi_rank)
grid = gridclass(N1,N2,N3,x,y,z,kc,num_processes,L1,L3,mpi_rank,comm,turb_model)
main = variables(grid,u,v,w,t,dt,et,nu,myFFT,Re_tau,turb_model,tau0,Cs,mpi_rank)
#====================================================================
main.computeStats = computeStats
main.iteration = iteration_start
main.save_freq = save_freq

### Main time integration loop
t0 = time.time()

#========================================================================
while (main.t < main.et):
  #------------- Save Output ------------------
  if (main.iteration%main.save_freq == 0):
    div = checkDivergence(main,grid)
    myFFT.myifft3D(main.uhat,main.u)
    myFFT.myifft3D(main.vhat,main.v)
    myFFT.myifft3D(main.what,main.w)
    divG = allGather_spectral(div,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npx)
    uGlobal = allGather_physical(main.u,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npy)
    vGlobal = allGather_physical(main.v,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npy)
    wGlobal = allGather_physical(main.w,comm,mpi_rank,grid.N1,grid.N2,grid.N3,num_processes,Npy)
    if (mpi_rank == 0):
      #string = '3DSolution/PVsol' + str(main.iteration)
      string2 = '3DSolution/npsol' + str(main.iteration)
      np.savez(string2,u=uGlobal,v=vGlobal,w=wGlobal)
      sys.stdout.write("===================================================================================== \n")
      sys.stdout.write('t = '  + str(main.t) + '   Wall time = ' + str(time.time() - t0) + '\n' )
      sys.stdout.write('Div = ' + str( np.linalg.norm(divG) )  + '\n')
  
      sys.stdout.flush()
      
      #gridToVTK(string, grid.xG,grid.yG,grid.zG, pointData = {"u" : np.real(uGlobal.transpose()) , \
      #  "v" : np.real(vGlobal.transpose()) , "w" : np.real(wGlobal.transpose()) } ) #, \
       #363       "p" : np.real(pdummy.transpose())} )

    ### Collect stats
    data1 = comm.gather(main.Ubar,root = 0)
    data2 = comm.gather(main.uubar,root = 0)
    if (mpi_rank == 0):
      Ubar_global = np.empty((3,N2))
      uubar_global = np.empty((6,N2))
      for j in range(0,num_processes):
        Ubar_global[ :,j*Npy:(j+1)*Npy] = data1[j][:,:]
        uubar_global[:,j*Npy:(j+1)*Npy] = data2[j][:,:]
    
      for i in range(0,3):
        Ubar_global[i] =   Ubar_global[i]/(main.save_iterations + 1.e-90)
      for i in range(0,6):
        uubar_global[i] = uubar_global[i]/(main.save_iterations + 1.e-90)
    
      rey_stress = np.zeros((6,grid.N2))
      rey_stress[0] = uubar_global[0] - Ubar_global[0]*Ubar_global[0]
      rey_stress[1] = uubar_global[1] - Ubar_global[1]*Ubar_global[1]
      rey_stress[2] = uubar_global[2] - Ubar_global[2]*Ubar_global[2]
      rey_stress[3] = uubar_global[3] - Ubar_global[0]*Ubar_global[1]
      rey_stress[4] = uubar_global[4] - Ubar_global[0]*Ubar_global[2]
      rey_stress[5] = uubar_global[5] - Ubar_global[1]*Ubar_global[2]
      string = '3DSolution/Stats' + str(main.iteration) 
      np.savez(string,U = Ubar_global,rey_stress = rey_stress,y=grid.yG[0,:,0])


  #---------------------------------------------
  # advance by Adams Bashforth/Crank Nicolson
  main.iteration += 1
  advance_AdamsCrank(main,grid,myFFT)
  main.t += main.dt
#========================================================================



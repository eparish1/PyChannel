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

nthreads = 1.
## Check if variables exist
#==============================================
if 'cfl' in globals():			     #|
  pass					     #|
else:				             #|
  cfl = -dt				     #|
#==============================================

# Make Solution Directory if it does not exist
if not os.path.exists('3DSolution'):
   os.makedirs('3DSolution')


# Initialize Classes. 
#=====================================================================
myFFT = FFTclass(N1,N2,N3,nthreads)
grid = gridclass(N1,N2,N3,x,y,z,kc)
main = variables(grid,u,v,w,t,dt,et,nu,myFFT,Re_tau)
#====================================================================

main.iteration = 0
main.save_freq = save_freq

### Main time integration loop
t0 = time.time()

ucheck = myFFT.myifft3D( myFFT.myfft3D(main.u))
checkVal = np.linalg.norm(main.u - ucheck) 
print('FFT CHECK = ' + str(checkVal) )
if ( checkVal >= 1e-10 ):
  sys.stdout.write('ERROR! FFT Check routines have error -> ifft(fft(u)) = ' + str(checkVal) + ' \n')
  sys.stdout.write('Quitting! \n')
  sys.stdout.flush()
  exit(0) 
#========================================================================
while (main.t < main.et):
  #------------- Save Output ------------------
  if (main.iteration%main.save_freq == 0):
    div = checkDivergence(main,grid)
    #print('Laminar Error = ' + str(norm(u - u_exact )))
    string = '3DSolution/PVsol' + str(main.iteration)
    string2 = '3DSolution/npsol' + str(main.iteration)
    main.u = myFFT.myifft3D(main.uhat)
    main.v = myFFT.myifft3D(main.vhat)
    main.w = myFFT.myifft3D(main.vhat)
    np.savez(string2,u=main.u,v=main.v,w=main.w)
    #main.p = myFFT.myifft3D(main.phat)
    sys.stdout.write("===================================================================================== \n")
    sys.stdout.write('t = '  + str(main.t) + '   Wall time = ' + str(time.time() - t0) + '\n' )
    sys.stdout.write('Div = ' + str( np.linalg.norm(div) )  + '\n')

    sys.stdout.flush()
    
    gridToVTK(string, x,y,z, pointData = {"u" : np.real(main.u.transpose()) , \
      "v" : np.real(main.v.transpose()) , "w" : np.real(main.w.transpose()) } ) #, \
#363       "p" : np.real(pdummy.transpose())} )

  #---------------------------------------------
  # advance by Adams Bashforth/Crank Nicolson
  main.iteration += 1
  advance_AdamsCrank(main,grid,myFFT)
  main.t += main.dt
#========================================================================



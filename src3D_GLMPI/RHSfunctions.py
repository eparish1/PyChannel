import threading
import scipy
import time
import sys
import scipy.sparse.linalg
import multiprocessing as mp
import numpy as np
from padding import separateModes
def allGather_physical(tmp_local,comm,mpi_rank,N1,N2,N3,num_processes,Npy):
  data = comm.gather(tmp_local,root = 0)
  if (mpi_rank == 0):
    tmp_global = np.empty((N1,N2,N3))
    for j in range(0,num_processes):
      tmp_global[:,j*Npy:(j+1)*Npy,:] = data[j][:,:,:]
    return tmp_global

def allGather_spectral(tmp_local,comm,mpi_rank,N1,N2,N3,num_processes,Npx):
  data = comm.gather(tmp_local,root = 0)
  if (mpi_rank == 0):
    tmp_global = np.empty((N1,N2,N3/2+1),dtype='complex')
    for j in range(0,num_processes):
      tmp_global[j*Npx:(j+1)*Npx,:,:] = data[j][:,:,:]
    return tmp_global


def getA1Mat(N2):
  A1 = np.zeros((N2,N2))
  for n in range(0,N2-1):
    for p in range(n+1,N2,2):
      A1[n,p] = 2.*p
  A1[0,:] = A1[0,:] / 2.
  return A1

def getA2Mat(N2):
  A2 = np.zeros((N2,N2))
  for n in range(0,N2-2):
    for p in range(n+2,N2,2):
      A2[n,p] = p*(p**2-n**2)
  A2[0,:] = A2[0,:] /2.
  return A2

def checkDivergence(main,grid):
  uhat_x = 1j*grid.k1*main.uhat
  vhat_y = diff_y(main.vhat)
  what_z = 1j*grid.k3*main.what
  div = (uhat_x[:,:,:] + vhat_y[:,:,:] + what_z[:,:,:])
  div[:,-1,:] = 0.
  return div

def diff_y(fhat):
  N1,N2,N3 = np.shape(fhat) 
  fhat1 = np.zeros((N1,N2,N3),dtype='complex') 
  for n in range(0,N2-1):
    for p in range(n+1,N2,2):
      fhat1[:,n,:] += fhat[:,p,:]*2.*p
  fhat1[:,0,:] = fhat1[:,0,:]/2.
  return fhat1

def diff_y2(uhat):
  N1,N2,N3 = np.shape(uhat)
  uhat2 = np.zeros((N1,N2,N3),dtype='complex')
  for n in range(0,N2-2):
    for p in range(n+2,N2,2):
      uhat2[:,n,:] += uhat[:,p,:]* p*(p**2 - n**2)
  uhat2[:,0,:] = uhat2[:,0,:]/2
  return uhat2


def getRHS_vort_FM1(main,grid,myFFT):
  main.uhat = grid.dealias_2x*main.uhat
  main.vhat = grid.dealias_2x*main.vhat
  main.what = grid.dealias_2x*main.what
  main.phat = grid.dealias_2x*main.phat

  u_pad = myFFT.myifft3D(main.uhat)
  v_pad = myFFT.myifft3D(main.vhat)
  w_pad = myFFT.myifft3D(main.what)

  ## compute vorticity
  omegahat_1 = diff_y(main.what) - 1j*grid.k3*main.vhat
  omegahat_2 = 1j*grid.k3*main.uhat - 1j*grid.k1*main.what
  omegahat_3 = 1j*grid.k1*main.vhat - diff_y(main.uhat)

  omega1_pad = myFFT.myifft3D(omegahat_1)
  omega2_pad = myFFT.myifft3D(omegahat_2)
  omega3_pad = myFFT.myifft3D(omegahat_3)

  uu_pad = u_pad*u_pad
  vv_pad = v_pad*v_pad
  ww_pad = w_pad*w_pad

  vom3_pad = v_pad*omega3_pad
  wom2_pad = w_pad*omega2_pad
  uom3_pad = u_pad*omega3_pad
  wom1_pad = w_pad*omega1_pad
  uom2_pad = u_pad*omega2_pad
  vom1_pad = v_pad*omega1_pad


  uuhat = myFFT.dealias_y( myFFT.myfft3D(uu_pad) )
  vvhat = myFFT.dealias_y( myFFT.myfft3D(vv_pad) )
  wwhat = myFFT.dealias_y( myFFT.myfft3D(ww_pad) )

  vom3_hat = myFFT.dealias_y( myFFT.myfft3D(vom3_pad)  )
  wom2_hat = myFFT.dealias_y( myFFT.myfft3D(wom2_pad)  )
  uom3_hat = myFFT.dealias_y( myFFT.myfft3D(uom3_pad)  )
  wom1_hat = myFFT.dealias_y( myFFT.myfft3D(wom1_pad)  )
  uom2_hat = myFFT.dealias_y( myFFT.myfft3D(uom2_pad)  )
  vom1_hat = myFFT.dealias_y( myFFT.myfft3D(vom1_pad)  )


  vsqrhat = 0.5*( uuhat + vvhat + wwhat)
  PLu = -( wom2_hat -vom3_hat + 1j*grid.k1*vsqrhat ) - main.dP  ### mean pressure gradient only
  PLv = -( uom3_hat -wom1_hat + diff_y(vsqrhat)    )
  PLw = -( vom1_hat -uom2_hat + 1j*grid.k3*vsqrhat )
 

  ## Now compute stuff for MZ!
  PLu_p, PLu_q = separateModes(PLu,1)
  PLv_p, PLv_q = separateModes(PLv,1)
  PLw_p, PLw_q = separateModes(PLw,1)

  PLu_qreal = myFFT.myifft3D(PLu_q)
  PLv_qreal = myFFT.myifft3D(PLv_q)
  PLw_qreal = myFFT.myifft3D(PLw_q)

  up_PLuq =  myFFT.myfft3D(u_pad*PLu_qreal)
  vp_PLuq =  myFFT.myfft3D(v_pad*PLu_qreal)
  wp_PLuq =  myFFT.myfft3D(w_pad*PLu_qreal)

  up_PLvq =  myFFT.myfft3D(u_pad*PLv_qreal)
  vp_PLvq =  myFFT.myfft3D(v_pad*PLv_qreal)
  wp_PLvq =  myFFT.myfft3D(w_pad*PLv_qreal)

  up_PLwq =  myFFT.myfft3D(u_pad*PLw_qreal)
  vp_PLwq =  myFFT.myfft3D(v_pad*PLw_qreal)
  wp_PLwq =  myFFT.myfft3D(w_pad*PLw_qreal)

  main.PLQLu =  -1j*grid.k1*up_PLuq - diff_y(vp_PLuq) - 1j*grid.k3*wp_PLuq - \
          1j*grid.k1*up_PLuq - diff_y(up_PLvq) - 1j*grid.k3*up_PLwq

  main.PLQLv =  -1j*grid.k1*up_PLvq - diff_y(vp_PLvq) - 1j*grid.k3*wp_PLvq - \
          1j*grid.k1*vp_PLuq - diff_y(vp_PLvq) - 1j*grid.k3*vp_PLwq

  main.PLQLw =  -1j*grid.k1*up_PLwq - diff_y(vp_PLwq) - 1j*grid.k3*wp_PLwq -\
          1j*grid.k1*wp_PLuq - diff_y(wp_PLvq) - 1j*grid.k3*wp_PLwq

  main.RHS_explicit[0] = PLu[:,:,:] + main.w0_u[:,:,:,0]
  main.RHS_explicit[1] = PLv[:,:,:] + main.w0_v[:,:,:,0]
  main.RHS_explicit[2] = PLw[:,:,:] + main.w0_w[:,:,:,0]
  main.RHS_explicit[3] = 2.*main.PLQLu
  main.RHS_explicit[4] = 2.*main.PLQLv
  main.RHS_explicit[5] = 2.*main.PLQLw



  uhat_xx = -grid.k1**2*main.uhat
  uhat_yy = diff_y2(main.uhat)
  uhat_zz = -grid.k3**2*main.uhat

  vhat_xx= -grid.k1**2*main.vhat
  vhat_yy= diff_y2(main.vhat)
  vhat_zz= -grid.k3**2*main.vhat

  what_xx= -grid.k1**2*main.what
  what_yy= diff_y2(main.what)
  what_zz= -grid.k3**2*main.what


  main.RHS_implicit[0] = main.nu*(uhat_xx + uhat_yy + uhat_zz) - 1j*grid.k1*main.phat
  main.RHS_implicit[1] = main.nu*(vhat_xx + vhat_yy + vhat_zz) - diff_y(main.phat)
  main.RHS_implicit[2] = main.nu*(what_xx + what_yy + what_zz) - 1j*grid.k3*main.phat
  main.RHS_implicit[3] = -2.*main.w0_u[:,:,:,0] / main.tau0
  main.RHS_implicit[4] = -2.*main.w0_v[:,:,:,0] / main.tau0
  main.RHS_implicit[5] = -2.*main.w0_w[:,:,:,0] / main.tau0


def getRHS_vort_Smag(main,grid,myFFT):
  main.uhat = grid.dealias*main.uhat
  main.vhat = grid.dealias*main.vhat
  main.what = grid.dealias*main.what
  main.phat = grid.dealias*main.phat

  u_pad = myFFT.myifft3D(main.uhat)
  v_pad = myFFT.myifft3D(main.vhat)
  w_pad = myFFT.myifft3D(main.what)

  ## compute vorticity
  omegahat_1 = diff_y(main.what) - 1j*grid.k3*main.vhat
  omegahat_2 = 1j*grid.k3*main.uhat - 1j*grid.k1*main.what
  omegahat_3 = 1j*grid.k1*main.vhat - diff_y(main.uhat)

  omega1_pad = myFFT.myifft3D(omegahat_1)
  omega2_pad = myFFT.myifft3D(omegahat_2)
  omega3_pad = myFFT.myifft3D(omegahat_3)

  uu_pad = u_pad*u_pad
  vv_pad = v_pad*v_pad
  ww_pad = w_pad*w_pad

  vom3_pad = v_pad*omega3_pad
  wom2_pad = w_pad*omega2_pad
  uom3_pad = u_pad*omega3_pad
  wom1_pad = w_pad*omega1_pad
  uom2_pad = u_pad*omega2_pad
  vom1_pad = v_pad*omega1_pad


  uuhat = grid.dealias*myFFT.myfft3D(uu_pad)
  vvhat = grid.dealias*myFFT.myfft3D(vv_pad)
  wwhat = grid.dealias*myFFT.myfft3D(ww_pad)
  vom3_hat = grid.dealias*myFFT.myfft3D(vom3_pad)
  wom2_hat = grid.dealias*myFFT.myfft3D(wom2_pad)
  uom3_hat = grid.dealias*myFFT.myfft3D(uom3_pad)
  wom1_hat = grid.dealias*myFFT.myfft3D(wom1_pad)
  uom2_hat = grid.dealias*myFFT.myfft3D(uom2_pad)
  vom1_hat = grid.dealias*myFFT.myfft3D(vom1_pad)

  ## Smagorinsky
  S11hat = 1j*grid.k1*main.uhat
  S22hat = diff_y(main.vhat)
  S33hat = 1j*grid.k3*main.what
  S12hat = 0.5*(diff_y(main.uhat) + 1j*grid.k1*main.vhat)
  S13hat = 0.5*(1j*grid.k3*main.uhat + 1j*grid.k1*main.what)
  S23hat = 0.5*(1j*grid.k3*main.vhat + diff_y(main.what) )

  S11real = np.zeros( (int(grid.N1),int(grid.Npy),int(grid.N3)) )
  S22real = np.zeros( (int(grid.N1),int(grid.Npy),int(grid.N3)) )
  S33real = np.zeros( (int(grid.N1),int(grid.Npy),int(grid.N3)) )
  S12real = np.zeros( (int(grid.N1),int(grid.Npy),int(grid.N3)) )
  S13real = np.zeros( (int(grid.N1),int(grid.Npy),int(grid.N3)) )
  S23real = np.zeros( (int(grid.N1),int(grid.Npy),int(grid.N3)) )

  S11real[:,:,:] = myFFT.myifft3D(S11hat)
  S22real[:,:,:] = myFFT.myifft3D(S22hat)
  S33real[:,:,:] = myFFT.myifft3D(S33hat)
  S12real[:,:,:] = myFFT.myifft3D(S12hat)
  S13real[:,:,:] = myFFT.myifft3D(S13hat)
  S23real[:,:,:] = myFFT.myifft3D(S23hat)

  S_magreal = np.sqrt( 2.*(S11real*S11real + S22real*S22real + S33real*S33real + \
            2.*S12real*S12real + 2.*S13real*S13real + 2.*S23real*S23real ) )
  nutreal = main.Delta[None,:,None]*main.Delta[None,:,None]*np.abs(S_magreal)

  tau11real = -2.*nutreal*S11real
  tau22real = -2.*nutreal*S22real
  tau33real = -2.*nutreal*S33real
  tau12real = -2.*nutreal*S12real
  tau13real = -2.*nutreal*S13real
  tau23real = -2.*nutreal*S23real

  tauhat = np.zeros((grid.Npx,grid.N2,grid.N3/2+1,6),dtype='complex')
  tauhat[:,:,:,0] = myFFT.myfft3D( -2.*nutreal*S11real )  #11
  tauhat[:,:,:,1] = myFFT.myfft3D( -2.*nutreal*S22real )  #22
  tauhat[:,:,:,2] = myFFT.myfft3D( -2.*nutreal*S33real )  #33
  tauhat[:,:,:,3] = myFFT.myfft3D( -2.*nutreal*S12real )  #12
  tauhat[:,:,:,4] = myFFT.myfft3D( -2.*nutreal*S13real )  #13
  tauhat[:,:,:,5] = myFFT.myfft3D( -2.*nutreal*S23real )  #23

  main.w0_u[:,:,:,0] = -1j*grid.k1*tauhat[:,:,:,0] - diff_y(tauhat[:,:,:,3]) - 1j*grid.k3*tauhat[:,:,:,4]
  main.w0_v[:,:,:,0] = -1j*grid.k1*tauhat[:,:,:,3] - diff_y(tauhat[:,:,:,1]) - 1j*grid.k3*tauhat[:,:,:,5]
  main.w0_w[:,:,:,0] = -1j*grid.k1*tauhat[:,:,:,4] - diff_y(tauhat[:,:,:,5]) - 1j*grid.k3*tauhat[:,:,:,2]

  vsqrhat = 0.5*( uuhat + vvhat + wwhat)

  main.RHS_explicit[0] = -( wom2_hat -vom3_hat + 1j*grid.k1*vsqrhat ) - main.dP + main.w0_u[:,:,:,0]
  main.RHS_explicit[1] = -( uom3_hat -wom1_hat + diff_y(vsqrhat)    )           + main.w0_v[:,:,:,0]
  main.RHS_explicit[2] = -( vom1_hat -uom2_hat + 1j*grid.k3*vsqrhat )           + main.w0_w[:,:,:,0]

  uhat_xx = -grid.k1**2*main.uhat
  uhat_yy = diff_y2(main.uhat)
  uhat_zz = -grid.k3**2*main.uhat

  vhat_xx= -grid.k1**2*main.vhat
  vhat_yy= diff_y2(main.vhat)
  vhat_zz= -grid.k3**2*main.vhat

  what_xx= -grid.k1**2*main.what
  what_yy= diff_y2(main.what)
  what_zz= -grid.k3**2*main.what


  main.RHS_implicit[0] = main.nu*(uhat_xx + uhat_yy + uhat_zz) - 1j*grid.k1*main.phat
  main.RHS_implicit[1] = main.nu*(vhat_xx + vhat_yy + vhat_zz) - diff_y(main.phat)
  main.RHS_implicit[2] = main.nu*(what_xx + what_yy + what_zz) - 1j*grid.k3*main.phat


def getRHS_vort_DSmag(main,grid,myFFT):
  main.uhat = grid.dealias*main.uhat
  main.vhat = grid.dealias*main.vhat
  main.what = grid.dealias*main.what
  main.phat = grid.dealias*main.phat

  u_pad = myFFT.myifft3D(main.uhat)
  v_pad = myFFT.myifft3D(main.vhat)
  w_pad = myFFT.myifft3D(main.what)

  ## compute vorticity
  omegahat_1 = diff_y(main.what) - 1j*grid.k3*main.vhat
  omegahat_2 = 1j*grid.k3*main.uhat - 1j*grid.k1*main.what
  omegahat_3 = 1j*grid.k1*main.vhat - diff_y(main.uhat)

  omega1_pad = myFFT.myifft3D(omegahat_1)
  omega2_pad = myFFT.myifft3D(omegahat_2)
  omega3_pad = myFFT.myifft3D(omegahat_3)

  uu_pad = u_pad*u_pad
  vv_pad = v_pad*v_pad
  ww_pad = w_pad*w_pad

  vom3_pad = v_pad*omega3_pad
  wom2_pad = w_pad*omega2_pad
  uom3_pad = u_pad*omega3_pad
  wom1_pad = w_pad*omega1_pad
  uom2_pad = u_pad*omega2_pad
  vom1_pad = v_pad*omega1_pad


  uuhat = grid.dealias*myFFT.myfft3D(uu_pad)
  vvhat = grid.dealias*myFFT.myfft3D(vv_pad)
  wwhat = grid.dealias*myFFT.myfft3D(ww_pad)
  vom3_hat = grid.dealias*myFFT.myfft3D(vom3_pad)
  wom2_hat = grid.dealias*myFFT.myfft3D(wom2_pad)
  uom3_hat = grid.dealias*myFFT.myfft3D(uom3_pad)
  wom1_hat = grid.dealias*myFFT.myfft3D(wom1_pad)
  uom2_hat = grid.dealias*myFFT.myfft3D(uom2_pad)
  vom1_hat = grid.dealias*myFFT.myfft3D(vom1_pad)


  #### Dynamic Smagorinsky Contribution
  ## First Need to compute Leonard Stress. Apply test filter at scale k_L
  ## k_c = DSmag_alpha*k_L
  #kL = main.kc/DSmag_alpha
  uhatF = grid.DSmag_Filter(main.uhat)          #!!
  vhatF = grid.DSmag_Filter(main.vhat)          #!!
  whatF = grid.DSmag_Filter(main.what)          #!!
  urealF = myFFT.myifft3D(uhatF)
  vrealF = myFFT.myifft3D(vhatF)
  wrealF = myFFT.myifft3D(whatF)
  uuhatF = grid.dealias*myFFT.myfft3D(urealF*urealF)
  vvhatF = grid.dealias*myFFT.myfft3D(vrealF*vrealF)
  wwhatF = grid.dealias*myFFT.myfft3D(wrealF*wrealF)
  uvhatF = grid.dealias*myFFT.myfft3D(urealF*vrealF)
  uwhatF = grid.dealias*myFFT.myfft3D(urealF*wrealF)
  vwhatF = grid.dealias*myFFT.myfft3D(vrealF*wrealF)

  uvhat = grid.dealias*myFFT.myfft3D(u_pad*v_pad)
  uwhat = grid.dealias*myFFT.myfft3D(u_pad*w_pad)
  vwhat = grid.dealias*myFFT.myfft3D(v_pad*w_pad)


  ## Make Leonard Stress Tensor
  Lhat = np.zeros((grid.Npx,grid.N2,(grid.N3/2+1),6),dtype='complex')
  Lhat[:,:,:,0] = grid.DSmag_Filter(uuhat) - uuhatF
  Lhat[:,:,:,1] = grid.DSmag_Filter(vvhat) - vvhatF
  Lhat[:,:,:,2] = grid.DSmag_Filter(wwhat) - wwhatF
  Lhat[:,:,:,3] = grid.DSmag_Filter(uvhat) - uvhatF
  Lhat[:,:,:,4] = grid.DSmag_Filter(uwhat) - uwhatF
  Lhat[:,:,:,5] = grid.DSmag_Filter(vwhat) - vwhatF
  ## Now compute the resolved stress tensor and the filte

  S11hat = 1j*grid.k1*main.uhat
  S22hat = diff_y(main.vhat)
  S33hat = 1j*grid.k3*main.what
  S12hat = 0.5*(diff_y(main.uhat) + 1j*grid.k1*main.vhat)
  S13hat = 0.5*(1j*grid.k3*main.uhat + 1j*grid.k1*main.what)
  S23hat = 0.5*(1j*grid.k3*main.vhat + diff_y(main.what) )

  S11real = myFFT.myifft3D(S11hat)
  S22real = myFFT.myifft3D(S22hat)
  S33real = myFFT.myifft3D(S33hat)
  S12real = myFFT.myifft3D(S12hat)
  S13real = myFFT.myifft3D(S13hat)
  S23real = myFFT.myifft3D(S23hat)

  S_magreal = np.sqrt( 2.*(S11real*S11real + S22real*S22real + S33real*S33real + \
            2.*S12real*S12real + 2.*S13real*S13real + 2.*S23real*S23real ) )

  ## Filtered Stress Tensor
  S11hatF = 1j*grid.k1*uhatF
  S22hatF = diff_y(vhatF)
  S33hatF = 1j*grid.k3*whatF
  S12hatF = 0.5*(diff_y(uhatF)    + 1j*grid.k1*vhatF)
  S13hatF = 0.5*(1j*grid.k3*uhatF + 1j*grid.k1*whatF)
  S23hatF = 0.5*(1j*grid.k3*vhatF + diff_y(whatF)   )
  S11realF = myFFT.myifft3D(S11hatF)
  S22realF = myFFT.myifft3D(S22hatF)
  S33realF = myFFT.myifft3D(S33hatF)
  S12realF = myFFT.myifft3D(S12hatF)
  S13realF = myFFT.myifft3D(S13hatF)
  S23realF = myFFT.myifft3D(S23hatF)
  S_magrealF = np.sqrt( 2.*(S11realF*S11realF + S22realF*S22realF + S33realF*S33realF + \
            2.*S12realF*S12realF + 2.*S13realF*S13realF + 2.*S23realF*S23realF ) )

  ## Now compute terms needed for M. Do pseudo-spectral for |S|Sij
  # First do for S at the test filter
  SFS11F = S_magrealF*S11realF
  SFS22F = S_magrealF*S22realF
  SFS33F = S_magrealF*S33realF
  SFS12F = S_magrealF*S12realF
  SFS13F = S_magrealF*S13realF
  SFS23F = S_magrealF*S23realF
  # Now do for resolved S. Apply test filter after transforming back to freq space
  SS11  = S_magreal*S11real
  SS22  = S_magreal*S22real
  SS33  = S_magreal*S33real
  SS12  = S_magreal*S12real
  SS13  = S_magreal*S13real
  SS23  = S_magreal*S23real
  SS11hatF = grid.DSmag_Filter(  myFFT.myfft3D( SS11 ) ) # don't need to dealias as well,
  SS22hatF = grid.DSmag_Filter(  myFFT.myfft3D( SS22 ) ) # filtering takes care of that and more
  SS33hatF = grid.DSmag_Filter(  myFFT.myfft3D( SS33 ) )
  SS12hatF = grid.DSmag_Filter(  myFFT.myfft3D( SS12 ) )
  SS13hatF = grid.DSmag_Filter(  myFFT.myfft3D( SS13 ) )
  SS23hatF = grid.DSmag_Filter(  myFFT.myfft3D( SS23 ) )
  SS11F = myFFT.myifft3D( SS11hatF )
  SS22F = myFFT.myifft3D( SS22hatF )
  SS33F = myFFT.myifft3D( SS33hatF )
  SS12F = myFFT.myifft3D( SS12hatF )
  SS13F = myFFT.myifft3D( SS13hatF )
  SS23F = myFFT.myifft3D( SS23hatF )

  DSmag_alpha = 2. # test filter at twice the true filter
  Mreal = np.zeros((grid.N1,grid.Npy,grid.N3,6))
  Mreal[:,:,:,0] = -2.*main.Delta[None,:,None]**2*(DSmag_alpha**2*SFS11F - SS11F)
  Mreal[:,:,:,1] = -2.*main.Delta[None,:,None]**2*(DSmag_alpha**2*SFS22F - SS22F)
  Mreal[:,:,:,2] = -2.*main.Delta[None,:,None]**2*(DSmag_alpha**2*SFS33F - SS33F)
  Mreal[:,:,:,3] = -2.*main.Delta[None,:,None]**2*(DSmag_alpha**2*SFS12F - SS12F)
  Mreal[:,:,:,4] = -2.*main.Delta[None,:,None]**2*(DSmag_alpha**2*SFS13F - SS13F)
  Mreal[:,:,:,5] = -2.*main.Delta[None,:,None]**2*(DSmag_alpha**2*SFS23F - SS23F)

  ## Now create Mhat tensor
  Mhat = np.zeros((grid.Npx,grid.N2,(grid.N3/2+1),6),dtype='complex')
#  print(np.shape(main.Delta),np.shape(SFS11Fhat))
#  Mhat[:,:,:,0] = -2.*main.Delta[None,:,None]**2*(DSmag_alpha**2*SFS11Fhat - SS11hatF)
#  Mhat[:,:,:,1] = -2.*main.Delta[None,:,None]**2*(DSmag_alpha**2*SFS22Fhat - SS22hatF)
#  Mhat[:,:,:,2] = -2.*main.Delta[None,:,None]**2*(DSmag_alpha**2*SFS33Fhat - SS33hatF)
#  Mhat[:,:,:,3] = -2.*main.Delta[None,:,None]**2*(DSmag_alpha**2*SFS12Fhat - SS12hatF)
#  Mhat[:,:,:,4] = -2.*main.Delta[None,:,None]**2*(DSmag_alpha**2*SFS13Fhat - SS13hatF)
#  Mhat[:,:,:,5] = -2.*main.Delta[None,:,None]**2*(DSmag_alpha**2*SFS23Fhat - SS23hatF)


  ## Now find Cs^2 = <Lij Mij>/<Mij Mij> Need to transform back to real space to get this
  #Mreal = np.zeros( (int(grid.N1),int(grid.Npy),int(grid.N3),6) )
  Lreal = np.zeros( (int(grid.N1),int(grid.Npy),int(grid.N3),6) )
  #Mreal[:,:,:,0] = myFFT.myifft3D(Mhat[:,:,:,0])
  #Mreal[:,:,:,1] = myFFT.myifft3D(Mhat[:,:,:,1])
  #Mreal[:,:,:,2] = myFFT.myifft3D(Mhat[:,:,:,2])
  #Mreal[:,:,:,3] = myFFT.myifft3D(Mhat[:,:,:,3])
  #Mreal[:,:,:,4] = myFFT.myifft3D(Mhat[:,:,:,4])
  #Mreal[:,:,:,5] = myFFT.myifft3D(Mhat[:,:,:,5])
  Lreal[:,:,:,0] = myFFT.myifft3D(Lhat[:,:,:,0])
  Lreal[:,:,:,1] = myFFT.myifft3D(Lhat[:,:,:,1])
  Lreal[:,:,:,2] = myFFT.myifft3D(Lhat[:,:,:,2])
  Lreal[:,:,:,3] = myFFT.myifft3D(Lhat[:,:,:,3])
  Lreal[:,:,:,4] = myFFT.myifft3D(Lhat[:,:,:,4])
  Lreal[:,:,:,5] = myFFT.myifft3D(Lhat[:,:,:,5])
  num = Mreal[:,:,:,0]*Lreal[:,:,:,0]  + Mreal[:,:,:,1]*Lreal[:,:,:,1] + Mreal[:,:,:,2]*Lreal[:,:,:,2] + \
         Mreal[:,:,:,3]*Lreal[:,:,:,3]  + Mreal[:,:,:,4]*Lreal[:,:,:,4] + Mreal[:,:,:,5]*Lreal[:,:,:,5]
  den = Mreal[:,:,:,0]*Mreal[:,:,:,0]  + Mreal[:,:,:,1]*Mreal[:,:,:,1] + Mreal[:,:,:,2]*Mreal[:,:,:,2] + \
         Mreal[:,:,:,3]*Mreal[:,:,:,3]  + Mreal[:,:,:,4]*Mreal[:,:,:,4] + Mreal[:,:,:,5]*Mreal[:,:,:,5]
  Cs_sqr = np.mean(np.mean(num,axis=2),axis=0)/(np.mean(np.mean(den,axis=2),axis=0) + 1.e-50)

  nutreal = Cs_sqr[None,:,None]*main.Delta[None,:,None]*main.Delta[None,:,None]*np.abs(S_magreal)


  tau11real = -2.*nutreal*S11real
  tau22real = -2.*nutreal*S22real
  tau33real = -2.*nutreal*S33real
  tau12real = -2.*nutreal*S12real
  tau13real = -2.*nutreal*S13real
  tau23real = -2.*nutreal*S23real

  tauhat = np.zeros((grid.Npx,grid.N2,grid.N3/2+1,6),dtype='complex')
  tauhat[:,:,:,0] = grid.dealias*myFFT.myfft3D( -2.*nutreal*S11real )  #11
  tauhat[:,:,:,1] = grid.dealias*myFFT.myfft3D( -2.*nutreal*S22real )  #22
  tauhat[:,:,:,2] = grid.dealias*myFFT.myfft3D( -2.*nutreal*S33real )  #33
  tauhat[:,:,:,3] = grid.dealias*myFFT.myfft3D( -2.*nutreal*S12real )  #12
  tauhat[:,:,:,4] = grid.dealias*myFFT.myfft3D( -2.*nutreal*S13real )  #13
  tauhat[:,:,:,5] = grid.dealias*myFFT.myfft3D( -2.*nutreal*S23real )  #23

  main.w0_u[:,:,:,0] = -1j*grid.k1*tauhat[:,:,:,0] - diff_y(tauhat[:,:,:,3]) - 1j*grid.k3*tauhat[:,:,:,4]
  main.w0_v[:,:,:,0] = -1j*grid.k1*tauhat[:,:,:,3] - diff_y(tauhat[:,:,:,1]) - 1j*grid.k3*tauhat[:,:,:,5]
  main.w0_w[:,:,:,0] = -1j*grid.k1*tauhat[:,:,:,4] - diff_y(tauhat[:,:,:,5]) - 1j*grid.k3*tauhat[:,:,:,2]

  vsqrhat = 0.5*( uuhat + vvhat + wwhat)

  main.RHS_explicit[0] = -( wom2_hat -vom3_hat + 1j*grid.k1*vsqrhat ) - main.dP + main.w0_u[:,:,:,0]
  main.RHS_explicit[1] = -( uom3_hat -wom1_hat + diff_y(vsqrhat)    )           + main.w0_v[:,:,:,0]
  main.RHS_explicit[2] = -( vom1_hat -uom2_hat + 1j*grid.k3*vsqrhat )           + main.w0_w[:,:,:,0]

  uhat_xx = -grid.k1**2*main.uhat
  uhat_yy = diff_y2(main.uhat)
  uhat_zz = -grid.k3**2*main.uhat

  vhat_xx= -grid.k1**2*main.vhat
  vhat_yy= diff_y2(main.vhat)
  vhat_zz= -grid.k3**2*main.vhat

  what_xx= -grid.k1**2*main.what
  what_yy= diff_y2(main.what)
  what_zz= -grid.k3**2*main.what


  main.RHS_implicit[0] = main.nu*(uhat_xx + uhat_yy + uhat_zz) - 1j*grid.k1*main.phat
  main.RHS_implicit[1] = main.nu*(vhat_xx + vhat_yy + vhat_zz) - diff_y(main.phat)
  main.RHS_implicit[2] = main.nu*(what_xx + what_yy + what_zz) - 1j*grid.k3*main.phat



def getRHS_vort(main,grid,myFFT):
  main.uhat = grid.dealias*main.uhat
  main.vhat = grid.dealias*main.vhat
  main.what = grid.dealias*main.what
  main.phat = grid.dealias*main.phat

  u = myFFT.myifft3D(main.uhat)
  v = myFFT.myifft3D(main.vhat)
  w = myFFT.myifft3D(main.what)

  ## compute vorticity
  omegahat_1 = diff_y(main.what) - 1j*grid.k3*main.vhat
  omegahat_2 = 1j*grid.k3*main.uhat - 1j*grid.k1*main.what
  omegahat_3 = 1j*grid.k1*main.vhat - diff_y(main.uhat)

  omega1 = myFFT.myifft3D(omegahat_1)
  omega2 = myFFT.myifft3D(omegahat_2)
  omega3 = myFFT.myifft3D(omegahat_3)

  uu = u*u
  vv = v*v
  ww = w*w

  vom3 = v*omega3
  wom2 = w*omega2
  uom3 = u*omega3
  wom1 = w*omega1
  uom2 = u*omega2
  vom1 = v*omega1


  uuhat = grid.dealias*( myFFT.myfft3D(uu) )
  vvhat = grid.dealias*( myFFT.myfft3D(vv) )
  wwhat = grid.dealias*( myFFT.myfft3D(ww) )

  vom3_hat = grid.dealias*( myFFT.myfft3D(vom3) )
  wom2_hat = grid.dealias*( myFFT.myfft3D(wom2) )
  uom3_hat = grid.dealias*( myFFT.myfft3D(uom3) )
  wom1_hat = grid.dealias*( myFFT.myfft3D(wom1) )
  uom2_hat = grid.dealias*( myFFT.myfft3D(uom2) )
  vom1_hat = grid.dealias*( myFFT.myfft3D(vom1) )


  vsqrhat = 0.5*( uuhat + vvhat + wwhat)
 
  main.RHS_explicit[0] = -( wom2_hat -vom3_hat + 1j*grid.k1*vsqrhat ) - main.dP ### mean pressure gradient only
  main.RHS_explicit[1] = -( uom3_hat -wom1_hat + diff_y(vsqrhat)    ) 
  main.RHS_explicit[2] = -( vom1_hat -uom2_hat + 1j*grid.k3*vsqrhat )  

 
  uhat_xx = -grid.k1**2*main.uhat
  uhat_yy = diff_y2(main.uhat)
  uhat_zz = -grid.k3**2*main.uhat

  vhat_xx= -grid.k1**2*main.vhat
  vhat_yy= diff_y2(main.vhat)
  vhat_zz= -grid.k3**2*main.vhat

  what_xx= -grid.k1**2*main.what
  what_yy= diff_y2(main.what)
  what_zz= -grid.k3**2*main.what


  main.RHS_implicit[0] = main.nu*(uhat_xx + uhat_yy + uhat_zz) - 1j*grid.k1*main.phat
  main.RHS_implicit[1] = main.nu*(vhat_xx + vhat_yy + vhat_zz) - diff_y(main.phat)
  main.RHS_implicit[2] = main.nu*(what_xx + what_yy + what_zz) - 1j*grid.k3*main.phat



def lineSolve(main,grid,myFFT,i,I,I2):
  altarray = (-np.ones(grid.N2))**(np.linspace(0,grid.N2-1,grid.N2))
  for k in range(0,grid.N3/2+1):
    if (grid.k1[i,0,k] == 0 and grid.k3[i,0,k] == 0):
      ### By continuity vhat = 0, don't need to solve for it.
      ### also by definition, p is the fixed pressure gradient, don't need to solve for it. 
      ### Just need to find u and w
      F = np.zeros((grid.N2,grid.N2),dtype='complex')
      F[:,:] =  -main.nu*( grid.A2[:,:]  )
      RHSu = np.zeros((grid.N2),dtype='complex')
      RHSu[:] = main.uhat[i,:,k] + main.dt/2.*(3.*main.RHS_explicit[0,i,:,k] - main.RHS_explicit_old[0,i,:,k]) + \
                main.dt/2.*( main.RHS_implicit[0,i,:,k] )
      RHSw = np.zeros((grid.N2),dtype='complex')
      RHSw[:] = main.what[i,:,k] + main.dt/2.*(3.*main.RHS_explicit[2,i,:,k] - main.RHS_explicit_old[2,i,:,k]) + \
                main.dt/2.*( main.RHS_implicit[2,i,:,k] )

      ## Now create entire LHS matrix
      LHSMAT = np.zeros((grid.N2,grid.N2),dtype='complex')
      ## insert in the linear contribution for the momentum equations
      LHSMAT[:,:] = np.eye(grid.N2) + 0.5*main.dt*F[:,:]
      ## Finally setup boundary condtions
      LHSMAT[-2,:] = 1.
      LHSMAT[-1,:] = altarray
      RHSu[-2::] = 0.
      RHSw[-2::] = 0.
      main.uhat[i,:,k] = np.linalg.solve(LHSMAT,RHSu)
      main.what[i,:,k] = np.linalg.solve(LHSMAT,RHSw)
      main.vhat[i,:,k] = 0. 
    else:
      if (abs(grid.k3[i,0,k]) <= grid.kcz): # don't bother solving for dealiased wave numbers
        t0 = time.time()
        ## SOLUTION VECTOR LOOKS LIKE
        #[ u0,v0,w0,ph0,u1,v1,w1,ph1,...,un,vn,wn]
        ## Form linear matrix for Crank Nicolson terms
        F = np.zeros((grid.N2*4-1,grid.N2*4-1),dtype='complex')
        #F = scipy.sparse.csc_matrix((grid.N2*4-1, grid.N2*4-1), dtype=complex).toarray()
        F[0::4,0::4] = -main.nu*( grid.A2[:,:] - grid.ksqr[i,0,k]*I2[:,:] )###Viscous terms
        F[1::4,1::4] = F[0::4,0::4]  ### put into v eqn as well
        F[2::4,2::4] = F[0::4,0::4]  ### put into w eqn as well
        np.fill_diagonal( F[0::4,3::4],1j*grid.k1[i,0,k] )  ## now add pressure to u eqn
        F[1:-2:4,3::4] = grid.A1p[:,:]                      ## v eqn
        np.fill_diagonal( F[2::4,3::4],1j*grid.k3[i,0,k] )  ## w eqn
   
        ## Now create RHS solution vector
        RHS = np.zeros((grid.N2*4-1),dtype='complex')
        RHS[0::4] = main.uhat[i,:,k] +  main.dt/2.*(3.*main.RHS_explicit[0,i,:,k] - main.RHS_explicit_old[0,i,:,k]) + main.dt/2.*main.RHS_implicit[0,i,:,k]
        RHS[1::4] = main.vhat[i,:,k] +  main.dt/2.*(3.*main.RHS_explicit[1,i,:,k] - main.RHS_explicit_old[1,i,:,k]) + main.dt/2.*main.RHS_implicit[1,i,:,k]
        RHS[2::4] = main.what[i,:,k] +  main.dt/2.*(3.*main.RHS_explicit[2,i,:,k] - main.RHS_explicit_old[2,i,:,k]) + main.dt/2.*main.RHS_implicit[2,i,:,k]
  
        LHSMAT = np.zeros((grid.N2*4-1,grid.N2*4-1),dtype='complex')
        #LHSMAT = scipy.sparse.csc_matrix((grid.N2*4-1, grid.N2*4-1), dtype=complex).toarray()
        ## insert in the linear contribution for the momentum equations
        LHSMAT[:,:] = I + 0.5*main.dt*F[:,:]
        ## Now add the continuity equation (evaluated at half grid points)
        LHSMAT[3::4,:] = 0.
        np.fill_diagonal( LHSMAT[3::4,0::4] , 1j*grid.k1[i,0,k] )  ##  du/dx
        LHSMAT[3::4,1::4] = grid.A1[0:-1,:]                        ##  dv/dy
        np.fill_diagonal( LHSMAT[3::4,2::4] , 1j*grid.k3[i,0,k] )  ##  dw/dz
        ## Finally setup boundary condtions
        LHSMAT[-7,:] = 0.
        LHSMAT[-6,:] = 0.
        LHSMAT[-5,:] = 0.
        LHSMAT[-3,:] = 0.
        LHSMAT[-2,:] = 0.
        LHSMAT[-1,:] = 0.
    
        LHSMAT[-7,0::4] = 1.
        LHSMAT[-6,1::4] = 1
        LHSMAT[-5,2::4] = 1
  
        LHSMAT[-3,0::4] = altarray[:]
        LHSMAT[-2,1::4] = altarray[:]
        LHSMAT[-1,2::4] = altarray[:]
  
        t1 = time.time() 
    #    solver = scipy.sparse.linalg.factorized( scipy.sparse.csc_matrix(LHSMAT))
    #    U = solver(RHS)
    #    U = np.linalg.solve(LHSMAT,RHS)
        U = (scipy.sparse.linalg.spsolve( scipy.sparse.csc_matrix(LHSMAT),RHS, permc_spec="NATURAL") )
    #    U = (scipy.sparse.linalg.bicgstab( scipy.sparse.csc_matrix(LHSMAT),RHS,tol=1e-14) )[0]
        main.uhat[i,:,k] = U[0::4]
        main.vhat[i,:,k] = U[1::4]
        main.what[i,:,k] = U[2::4]
        main.phat[i,0:-1,k] = U[3::4]
        main.LHSMAT = LHSMAT
        main.RHS = RHS
  


def solveBlock(main,grid,myFFT,I,I2,i_start,i_end):
  for i in range(0,grid.Npx):
    if (abs(grid.k1[i,0,0]) <= grid.kcx): #don't bother solving for dealiased wave numbers
       lineSolve(main,grid,myFFT,i,I,I2)



def advance_AdamsCrank(main,grid,myFFT):
  main.RHS_explicit_old[:,:,:] = main.RHS_explicit[:,:,:]
  t1 = time.time() 
  main.getRHS(main,grid,myFFT)
  I = np.eye(grid.N2*4-1)
  I2 = np.eye(grid.N2)
  t2 = time.time()
  solveBlock(main,grid,myFFT,I,I2,0,grid.N1)
  if (main.turb_model == 'FM1'):
    main.w0_u[:,:,:,0] =  (main.w0_u[:,:,:,0] + main.dt/2.*(3.*main.RHS_explicit[3] - \
                          main.RHS_explicit_old[3]) + main.dt/2.*main.RHS_implicit[3] )\
                          *main.tau0/(main.dt + main.tau0)
    main.w0_v[:,:,:,0] =  (main.w0_v[:,:,:,0] + main.dt/2.*(3.*main.RHS_explicit[4] - \
                          main.RHS_explicit_old[4]) + main.dt/2.*main.RHS_implicit[4] )\
                          *main.tau0/(main.dt + main.tau0)
    main.w0_w[:,:,:,0] =  (main.w0_w[:,:,:,0] + main.dt/2.*(3.*main.RHS_explicit[5] - \
                          main.RHS_explicit_old[5]) + main.dt/2.*main.RHS_implicit[5] )\
                          *main.tau0/(main.dt + main.tau0)

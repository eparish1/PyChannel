import threading
import scipy
import time
import sys
import scipy.sparse.linalg
import multiprocessing as mp
import numpy as np
from padding import separateModes
from mpi4py import MPI
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
  div = np.zeros(np.shape(main.uhat) ,dtype='complex')
  div[:,0:grid.kcy_int-1,:] = (uhat_x[:,0:grid.kcy_int-1,:] + vhat_y[:,0:grid.kcy_int-1,:] + what_z[:,0:grid.kcy_int-1,:])
  #div[:,0:grid.N2/2-1,:] = (uhat_x[:,0:grid.N2/2-1,:] + vhat_y[:,0:grid.N2/2-1,:] + what_z[:,0:grid.N2/2-1,:])
  div[:,-1,:] = 0.
  return div

def diff_y(fhat):
  N1,N2,N3 = np.shape(fhat)
  fhat1 = np.zeros((N1,N2,N3),dtype='complex')
  #fhat1[:,N2-1,:] = 0  k = N2-1
  fhat1[:,-2,:] = 2.*(N2-1)*fhat[:,-1,:] #k = N2-2
  for k in range(N2-3,-1,-1):
    fhat1[:,k,:] = fhat1[:,k+2,:] + 2.*(k+1)*fhat[:,k+1,:]
  fhat1[:,0,:] = fhat1[:,0,:]/2.
  return fhat1

def diff_y2(fhat):
  N1,N2,N3 = np.shape(fhat)
  fhat1 = np.zeros((N1,N2,N3),dtype='complex')
  fhat2 = np.zeros((N1,N2,N3),dtype='complex')
  #fhat1[:,N2-1,:] = 0  k = N2-1
  fhat1[:,-2,:] = 2.*(N2-1)*fhat[:,-1,:] #k = N2-2
  for k in range(N2-3,-1,-1):
    fhat1[:,k,:] = fhat1[:,k+2,:] + 2.*(k+1)*fhat[:,k+1,:]
  fhat1[:,0,:] = fhat1[:,0,:]/2.
  fhat2[:,-3,:] = 2.*(N2-2)*fhat1[:,-2,:] #k = N2-3
  for k in range(N2-4,-1,-1):
    fhat2[:,k,:] = fhat2[:,k+2,:] + 2.*(k+1)*fhat1[:,k+1,:]
  fhat2[:,0,:] = fhat2[:,0,:]/2.
  return fhat2


def getRHS_vort_dtau(main,grid,myFFT):
  main.uhat = grid.dealias_2x*main.uhat
  main.vhat = grid.dealias_2x*main.vhat
  main.what = grid.dealias_2x*main.what
  main.phat = grid.dealias_2x*main.phat

  def computePLU(uhat,vhat,what):
    u = myFFT.myifft3D(uhat)
    v = myFFT.myifft3D(vhat)
    w = myFFT.myifft3D(what)

    omegahat_1 = diff_y(what) - 1j*grid.k3*vhat
    omegahat_2 = 1j*grid.k3*uhat - 1j*grid.k1*what
    omegahat_3 = 1j*grid.k1*vhat - diff_y(uhat)

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

    uuhat = myFFT.dealias_y( myFFT.myfft3D(u*u) )
    vvhat = myFFT.dealias_y( myFFT.myfft3D(v*v) )
    wwhat = myFFT.dealias_y( myFFT.myfft3D(w*w) )
    #uvhat = myFFT.dealias_y( myFFT.myfft3D(u*v) )
    #uwhat = myFFT.dealias_y( myFFT.myfft3D(u*w) )
    #vwhat = myFFT.dealias_y( myFFT.myfft3D(v*w) )

    vom3_hat = myFFT.dealias_y( myFFT.myfft3D(vom3)  )
    wom2_hat = myFFT.dealias_y( myFFT.myfft3D(wom2)  )
    uom3_hat = myFFT.dealias_y( myFFT.myfft3D(uom3)  )
    wom1_hat = myFFT.dealias_y( myFFT.myfft3D(wom1)  )
    uom2_hat = myFFT.dealias_y( myFFT.myfft3D(uom2)  )
    vom1_hat = myFFT.dealias_y( myFFT.myfft3D(vom1)  )

    vsqrhat = 0.5*( uuhat + vvhat + wwhat)
    PLu = myFFT.dealias_y(  -( wom2_hat -vom3_hat + 1j*grid.k1*vsqrhat ) - main.dP ) ### mean pressure gradient only
    PLv = myFFT.dealias_y( -( uom3_hat -wom1_hat + diff_y(vsqrhat)    )             )
    PLw = myFFT.dealias_y( -( vom1_hat -uom2_hat + 1j*grid.k3*vsqrhat )             )

    #PLu = myFFT.dealias_y( -1j*grid.k1*uuhat - diff_y(uvhat) - 1j*grid.k3*uwhat - main.dP    ) ### mean pressure gradient only
    #PLv = myFFT.dealias_y( -1j*grid.k1*uvhat - diff_y(vvhat) - 1j*grid.k3*vwhat           )
    #PLw = myFFT.dealias_y( -1j*grid.k1*uwhat - diff_y(vwhat) - 1j*grid.k3*wwhat     )

    return u,v,w,PLu,PLv,PLw

  def computePLQLU(u,v,w,PLu,PLv,PLw,split_modes_mat):
    ## Now compute stuff for MZ!
    PLu_p, PLu_q = separateModes(PLu,split_modes_mat)
    PLv_p, PLv_q = separateModes(PLv,split_modes_mat)
    PLw_p, PLw_q = separateModes(PLw,split_modes_mat)

    PLu_qreal = myFFT.myifft3D(PLu_q)
    PLv_qreal = myFFT.myifft3D(PLv_q)
    PLw_qreal = myFFT.myifft3D(PLw_q)

    up_PLuq =  myFFT.myfft3D(u*PLu_qreal)
    vp_PLuq =  myFFT.myfft3D(v*PLu_qreal)
    wp_PLuq =  myFFT.myfft3D(w*PLu_qreal)

    up_PLvq =  myFFT.myfft3D(u*PLv_qreal)
    vp_PLvq =  myFFT.myfft3D(v*PLv_qreal)
    wp_PLvq =  myFFT.myfft3D(w*PLv_qreal)

    up_PLwq =  myFFT.myfft3D(u*PLw_qreal)
    vp_PLwq =  myFFT.myfft3D(v*PLw_qreal)
    wp_PLwq =  myFFT.myfft3D(w*PLw_qreal)

    PLQLu = -1j*grid.k1*up_PLuq - diff_y(vp_PLuq) - 1j*grid.k3*wp_PLuq - \
            1j*grid.k1*up_PLuq - diff_y(up_PLvq) - 1j*grid.k3*up_PLwq 
    PLQLv = -1j*grid.k1*up_PLvq - diff_y(vp_PLvq) - 1j*grid.k3*wp_PLvq - \
            1j*grid.k1*vp_PLuq - diff_y(vp_PLvq) - 1j*grid.k3*vp_PLwq 
    PLQLw = -1j*grid.k1*up_PLwq - diff_y(vp_PLwq) - 1j*grid.k3*wp_PLwq -\
            1j*grid.k1*wp_PLuq - diff_y(wp_PLvq) - 1j*grid.k3*wp_PLwq 
    return PLQLu,PLQLv,PLQLw

  u,v,w,PLu,PLv,PLw = computePLU(grid.dealias_2x*main.uhat,grid.dealias_2x*main.vhat,grid.dealias_2x*main.what) 
  PLQLu,PLQLv,PLQLw = computePLQLU(u,v,w,PLu,PLv,PLw,grid.dealias_2x)

  ## Now do dynamic procedure to get tau
  uhat_filt = grid.test_filter*main.uhat
  vhat_filt = grid.test_filter*main.vhat
  what_filt = grid.test_filter*main.what
  uf,vf,wf,P2Lu,P2Lv,P2Lw = computePLU(uhat_filt,vhat_filt,what_filt)
  P2LQLu,P2LQLv,P2LQLw = computePLQLU(uf,vf,wf,P2Lu,P2Lv,P2Lw,grid.test_filter)

  ## Now compute Leonard Stress
  #L11 = grid.test_filter*myFFT.myfft3D(u*u) - grid.test_filter*myFFT.myfft3D(uf*uf)
  #L22 = grid.test_filter*myFFT.myfft3D(v*v) - grid.test_filter*myFFT.myfft3D(vf*vf)
  #L33 = grid.test_filter*myFFT.myfft3D(w*w) - grid.test_filter*myFFT.myfft3D(wf*wf)
  #L12 = grid.test_filter*myFFT.myfft3D(u*v) - grid.test_filter*myFFT.myfft3D(uf*vf)
  #L13 = grid.test_filter*myFFT.myfft3D(u*w) - grid.test_filter*myFFT.myfft3D(uf*wf)
  #L23 = grid.test_filter*myFFT.myfft3D(v*w) - grid.test_filter*myFFT.myfft3D(vf*wf)

  #Lu1 = -1j*grid.k1*L11 - diff_y(L12) - 1j*grid.k3*L13
  #Lv = -1j*grid.k1*L12 - diff_y(L22) - 1j*grid.k3*L23
  #Lw = -1j*grid.k1*L13 - diff_y(L23) - 1j*grid.k3*L33

  Lu = grid.test_filter*(P2Lu - PLu)
  Lv = grid.test_filter*(P2Lv - PLv)
  Lw = grid.test_filter*(P2Lw - PLw)

  #uf = myFFT.myifft3D(grid.test_filter*main.uhat)
  #vf = myFFT.myifft3D(grid.test_filter*main.vhat)
  #wf = myFFT.myifft3D(grid.test_filter*main.what)

  # transfrom to real space to get tau
  #LuR = myFFT.myifft3D(grid.test_filter*Lu)
  #LvR = myFFT.myifft3D(grid.test_filter*Lv)
  #LwR = myFFT.myifft3D(grid.test_filter*Lw)
  #LE = np.mean((uf)*(LuR) + (vf)*(LvR) + (wf)*(LwR),axis=(0,2))
  #PLQLuR = myFFT.myifft3D(grid.test_filter*PLQLu)
  #PLQLvR = myFFT.myifft3D(grid.test_filter*PLQLv)
  #PLQLwR = myFFT.myifft3D(grid.test_filter*PLQLw)
  #PLQLE = np.mean((uf)*(PLQLuR) +(vf)*(PLQLvR) + (wf)*(PLQLwR),axis=(0,2))
#
#  P2LQLuR = myFFT.myifft3D(grid.test_filter*P2LQLu)
#  P2LQLvR = myFFT.myifft3D(grid.test_filter*P2LQLv)
#  P2LQLwR = myFFT.myifft3D(grid.test_filter*P2LQLw)
#  P2LQLE = np.mean((uf)*(P2LQLuR) + (vf)*(P2LQLvR) + (wf)*(P2LQLwR),axis=(0,2))
#
#  tau =  (LE / (PLQLE - P2LQLE) )
#  #plot(tau)
#  #pause(0.01)
  comm = MPI.COMM_WORLD
  num_processes = comm.Get_size()
  mpi_rank = comm.Get_rank()
##  tau_loc = comm.gather(tau,root = 0)
#  if (mpi_rank == 0):
#    tau_total = np.zeros((grid.N2))
#    for j in range(0,num_processes):
#      tau_total[j*grid.Npy:(j+1)*grid.Npy] = tau_loc[j]
#    np.savez('3DSolution/tau',tau=tau_total)
#    tau_mean = np.mean(tau_total)
#    sys.stdout.write(str(tau_mean) + "\n")
#    sys.stdout.flush()
#    #plot(tau_total)
#    #pause(0.01)
#    #clf()
#
#    for j in range(1,num_processes):
#      comm.send(tau_mean, dest=j)
#  else:
#    tau_mean = comm.recv(source=0)

  #print(tau_total)
  # go to cheb space
#  taumod = np.zeros(2*(grid.N2-1))
#  taumod[0:grid.N2] = tau[0:grid.N2]
#  taumod[grid.N2:2*(grid.N2-1)] = np.flipud(tau)[1:-1]
#  wtilde = np.fft.ifft(taumod) ## yes! actually the ifft. again, only god knows why
#  tauhat = np.zeros(grid.N2,dtype='complex')
#  tauhat[0] = wtilde[0]
#  tauhat[1:-1] = wtilde[1:grid.N2-1]*2.
#  tauhat[-1] = wtilde[grid.N2-1]
#
  #utau = main.Re_tau*main.nu 
  #plot(grid.y[0,:,0]*utau/main.nu + (utau/main.nu),tau)
  #ylim([-1,1])
  #pause(0.001)
  #tau = np.mean(tau)
  #print(tau)
  ## Now compute energy up to test filter
  LE =(np.sum(Lu[:,:,1:grid.N3/2]*np.conj(uhat_filt[:,:,1:grid.N3/2]*2),axis=(0,1,2) ) + \
       np.sum(Lu[:,:,0]*np.conj(uhat_filt[:,:,0]),axis=(0,1)) + \
       np.sum(Lv[:,:,1:grid.N3/2]*np.conj(vhat_filt[:,:,1:grid.N3/2]*2),axis=(0,1,2) ) + \
       np.sum(Lv[:,:,0]*np.conj(vhat_filt[:,:,0]),axis=(0,1)) + \
       np.sum(Lw[:,:,1:grid.N3/2]*np.conj(what_filt[:,:,1:grid.N3/2]*2),axis=(0,1,2) ) + \
       np.sum(Lw[:,:,0]*np.conj(what_filt[:,:,0]),axis=(0,1)) )
#
  PLQLE =(np.sum(PLQLu[:,:,1:grid.N3/2]*np.conj(uhat_filt[:,:,1:grid.N3/2]*2),axis=(0,1,2) ) + \
       np.sum(PLQLu[:,:,0]*np.conj(uhat_filt[:,:,0]),axis=(0,1)) + \
       np.sum(PLQLv[:,:,1:grid.N3/2]*np.conj(vhat_filt[:,:,1:grid.N3/2]*2),axis=(0,1,2) ) + \
       np.sum(PLQLv[:,:,0]*np.conj(vhat_filt[:,:,0]),axis=(0,1)) + \
       np.sum(PLQLw[:,:,1:grid.N3/2]*np.conj(what_filt[:,:,1:grid.N3/2]*2),axis=(0,1,2) ) + \
       np.sum(PLQLw[:,:,0]*np.conj(what_filt[:,:,0]),axis=(0,1)) )
#
  P2LQLE =(np.sum(P2LQLu[:,:,1:grid.N3/2]*np.conj(uhat_filt[:,:,1:grid.N3/2]*2),axis=(0,1,2) ) + \
       np.sum(P2LQLu[:,:,0]*np.conj(uhat_filt[:,:,0]),axis=(0,1)) + \
       np.sum(P2LQLv[:,:,1:grid.N3/2]*np.conj(vhat_filt[:,:,1:grid.N3/2]*2),axis=(0,1,2) ) + \
       np.sum(P2LQLv[:,:,0]*np.conj(vhat_filt[:,:,0]),axis=(0,1)) + \
       np.sum(P2LQLw[:,:,1:grid.N3/2]*np.conj(what_filt[:,:,1:grid.N3/2]*2),axis=(0,1,2) ) + \
       np.sum(P2LQLw[:,:,0]*np.conj(what_filt[:,:,0]),axis=(0,1)) )

  LE_loc = comm.gather(LE,root = 0)
  if (mpi_rank == 0):
    LE_total = 0 + 0j#np.zeros((grid.N2),dtype='complex')
    for j in range(0,num_processes):
      LE_total += LE_loc[j]
    for j in range(1,num_processes):
      comm.send(LE_total, dest=j)
  else:
    LE_total = comm.recv(source=0)

  PLQLE_loc = comm.gather(PLQLE,root = 0)
  if (mpi_rank == 0):
    PLQLE_total =  0 + 0j#np.zeros((grid.N2),dtype='complex')
    for j in range(0,num_processes):
      PLQLE_total += PLQLE_loc[j]
    for j in range(1,num_processes):
      comm.send(PLQLE_total, dest=j)
  else:
    PLQLE_total = comm.recv(source=0)

  P2LQLE_loc = comm.gather(P2LQLE,root = 0)
  if (mpi_rank == 0):
    P2LQLE_total =  0 + 0j #np.zeros((grid.N2),dtype='complex')
    for j in range(0,num_processes):
      P2LQLE_total += P2LQLE_loc[j]
    for j in range(1,num_processes):
      comm.send(P2LQLE_total, dest=j)
  else:
    P2LQLE_total = comm.recv(source=0)

#
#  N2 = grid.N2
  tau =  np.real( LE_total ) / (np.real(PLQLE_total)  - np.real(P2LQLE_total) + 1e-60 )
  if (mpi_rank == 0):
    print('tau ' + str(tau))
  #utau = main.Re_tau*main.nu 
  #plot(grid.y[0,:,0]*utau/main.nu + (utau/main.nu),tau)
  #ylim([-1,1])

#  #taumod = np.zeros(2*(grid.N2-1),dtype='complex')
  #taumod[0,] = tau[0]
  #taumod[1:N2] = tau[1::]/2
  #taumod[grid.N2::] = np.flipud(tau)[1:-1]/2.
  #taureal = np.fft.fft(taumod)[0:grid.N2] ##yes! actually the FFT! only god knows why

  #tau = np.clip(tau,0,2)
  #print(tau)#,np.real(LE[0:N2*2/3]),np.real(P2LQLE[0:N2*2/3]),np.real(PLQLE[0:N2*2/3])) 
  #plot(real(taureal))
  #ylim([-20,20])
  #tau = 0.1
  #main.w0_u[:,:,:,0] = myFFT.myfft3D(tau[None,:,None]*myFFT.myifft3D(PLQLu))
  #main.w0_v[:,:,:,0] = myFFT.myfft3D(tau[None,:,None]*myFFT.myifft3D(PLQLv))
  #main.w0_w[:,:,:,0] = myFFT.myfft3D(tau[None,:,None]*myFFT.myifft3D(PLQLw))

  main.w0_u[:,:,:,0] = tau*PLQLu
  main.w0_v[:,:,:,0] = tau*PLQLv
  main.w0_w[:,:,:,0] = tau*PLQLw

  main.RHS_explicit[0] = PLu[:,:,:] + main.w0_u[:,:,:,0]
  main.RHS_explicit[1] = PLv[:,:,:] + main.w0_v[:,:,:,0]
  main.RHS_explicit[2] = PLw[:,:,:] + main.w0_w[:,:,:,0]
  

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




def getRHS_vort_stau(main,grid,myFFT):
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
  PLu = myFFT.dealias_y(  -( wom2_hat -vom3_hat + 1j*grid.k1*vsqrhat ) - main.dP ) ### mean pressure gradient only
  PLv = myFFT.dealias_y( -( uom3_hat -wom1_hat + diff_y(vsqrhat)    )             )
  PLw = myFFT.dealias_y( -( vom1_hat -uom2_hat + 1j*grid.k3*vsqrhat )             )
 

  ## Now compute stuff for MZ!
  PLu_p, PLu_q = separateModes(PLu,grid.dealias_2x)
  PLv_p, PLv_q = separateModes(PLv,grid.dealias_2x)
  PLw_p, PLw_q = separateModes(PLw,grid.dealias_2x)

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

  main.w0_u[:,:,:,0] = main.tau0*( -1j*grid.k1*up_PLuq - diff_y(vp_PLuq) - 1j*grid.k3*wp_PLuq - \
          1j*grid.k1*up_PLuq - diff_y(up_PLvq) - 1j*grid.k3*up_PLwq )
  main.w0_v[:,:,:,0] = main.tau0*( -1j*grid.k1*up_PLvq - diff_y(vp_PLvq) - 1j*grid.k3*wp_PLvq - \
          1j*grid.k1*vp_PLuq - diff_y(vp_PLvq) - 1j*grid.k3*vp_PLwq )

  main.w0_w[:,:,:,0] = main.tau0*( -1j*grid.k1*up_PLwq - diff_y(vp_PLwq) - 1j*grid.k3*wp_PLwq -\
          1j*grid.k1*wp_PLuq - diff_y(wp_PLvq) - 1j*grid.k3*wp_PLwq )

  main.RHS_explicit[0] = PLu[:,:,:] + main.w0_u[:,:,:,0]
  main.RHS_explicit[1] = PLv[:,:,:] + main.w0_v[:,:,:,0]
  main.RHS_explicit[2] = PLw[:,:,:] + main.w0_w[:,:,:,0]
  

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
  PLu = myFFT.dealias_y(  -( wom2_hat -vom3_hat + 1j*grid.k1*vsqrhat ) - main.dP ) ### mean pressure gradient only
  PLv = myFFT.dealias_y( -( uom3_hat -wom1_hat + diff_y(vsqrhat)    )             )
  PLw = myFFT.dealias_y( -( vom1_hat -uom2_hat + 1j*grid.k3*vsqrhat )             )
 

  ## Now compute stuff for MZ!
  PLu_p, PLu_q = separateModes(PLu,grid.dealias_2x)
  PLv_p, PLv_q = separateModes(PLv,grid.dealias_2x)
  PLw_p, PLw_q = separateModes(PLw,grid.dealias_2x)

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
  SFS11F = abs( S_magrealF )*S11realF
  SFS22F = abs( S_magrealF )*S22realF
  SFS33F = abs( S_magrealF )*S33realF
  SFS12F = abs( S_magrealF )*S12realF
  SFS13F = abs( S_magrealF )*S13realF
  SFS23F = abs( S_magrealF )*S23realF
  # Now do for resolved S. Apply test filter after transforming back to freq space
  SS11  = abs( S_magreal )*S11real
  SS22  = abs( S_magreal )*S22real
  SS33  = abs( S_magreal )*S33real
  SS12  = abs( S_magreal )*S12real
  SS13  = abs( S_magreal )*S13real
  SS23  = abs( S_magreal )*S23real
  SS11hatF = grid.DSmag_Filter( grid.dealias* myFFT.myfft3D( SS11 ) ) # don't need to dealias as well,
  SS22hatF = grid.DSmag_Filter(grid.dealias*  myFFT.myfft3D( SS22 ) ) # filtering takes care of that and more
  SS33hatF = grid.DSmag_Filter( grid.dealias* myFFT.myfft3D( SS33 ) )
  SS12hatF = grid.DSmag_Filter( grid.dealias* myFFT.myfft3D( SS12 ) )
  SS13hatF = grid.DSmag_Filter( grid.dealias* myFFT.myfft3D( SS13 ) )
  SS23hatF = grid.DSmag_Filter( grid.dealias* myFFT.myfft3D( SS23 ) )
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
         2.*Mreal[:,:,:,3]*Lreal[:,:,:,3]  + 2.*Mreal[:,:,:,4]*Lreal[:,:,:,4] + 2.*Mreal[:,:,:,5]*Lreal[:,:,:,5]
  den = Mreal[:,:,:,0]*Mreal[:,:,:,0]  + Mreal[:,:,:,1]*Mreal[:,:,:,1] + Mreal[:,:,:,2]*Mreal[:,:,:,2] + \
         2.*Mreal[:,:,:,3]*Mreal[:,:,:,3]  + 2.*Mreal[:,:,:,4]*Mreal[:,:,:,4] + 2.*Mreal[:,:,:,5]*Mreal[:,:,:,5]
  Cs_sqr = np.mean(np.mean(num,axis=2),axis=0)/(np.mean(np.mean(den,axis=2),axis=0) + 1.e-50)

  nutreal = Cs_sqr[None,:,None]*main.Delta[None,:,None]*main.Delta[None,:,None]*np.abs(S_magreal)
  comm = MPI.COMM_WORLD
  num_processes = comm.Get_size()
  mpi_rank = comm.Get_rank()
  #if (mpi_rank == 0):
  #  sys.stdout.write(str(Cs_sqr*main.Delta[:]**2) + '\n')
  #  sys.stdout.flush()
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

## This has the tau(y). It is computed by getting the stresses and computing tau in the chebyshev modes
def getRHS_vort_dtau_3(main,grid,myFFT):
  def evalRHS(Uhat,grid,myFFT,main):
    U = np.zeros((3,grid.N1,grid.Npy,grid.N3))
    omega = np.zeros((3,grid.N1,grid.Npy,grid.N3))
    omegahat = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    U[0] = myFFT.myifft3D(Uhat[0])
    U[1] = myFFT.myifft3D(Uhat[1])
    U[2] = myFFT.myifft3D(Uhat[2])
  
    ## compute vorticity
    omegahat[0] = diff_y(Uhat[2]) - 1j*grid.k3*Uhat[1]
    omegahat[1] = 1j*grid.k3*Uhat[0] - 1j*grid.k1*Uhat[2]
    omegahat[2] = 1j*grid.k1*Uhat[1] - diff_y(Uhat[0])
 
    omega[0] = myFFT.myifft3D(omegahat[0])
    omega[1] = myFFT.myifft3D(omegahat[1])
    omega[2] = myFFT.myifft3D(omegahat[2])
  
    uu = U[0]*U[0]
    vv = U[1]*U[1]
    ww = U[2]*U[2]
  
    vom3 = U[1]*omega[2]
    wom2 = U[2]*omega[1]
    uom3 = U[0]*omega[2]
    wom1 = U[2]*omega[0]
    uom2 = U[0]*omega[1]
    vom1 = U[1]*omega[0]
  
  
    uuhat = myFFT.myfft3D(uu) 
    vvhat = myFFT.myfft3D(vv) 
    wwhat = myFFT.myfft3D(ww) 
  
    vom3_hat = myFFT.myfft3D(vom3) 
    wom2_hat = myFFT.myfft3D(wom2) 
    uom3_hat = myFFT.myfft3D(uom3) 
    wom1_hat = myFFT.myfft3D(wom1) 
    uom2_hat = myFFT.myfft3D(uom2) 
    vom1_hat = myFFT.myfft3D(vom1) 
  
    vsqrhat = 0.5*( uuhat + vvhat + wwhat)

    RHS_explicit = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    RHS_implicit = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    del2_Uhat = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')

    RHS_explicit[0] = -( wom2_hat -vom3_hat + 1j*grid.k1*vsqrhat ) - main.dP ### mean pressure gradient only
    RHS_explicit[1] = -( uom3_hat -wom1_hat + diff_y(vsqrhat)    )
    RHS_explicit[2] = -( vom1_hat -uom2_hat + 1j*grid.k3*vsqrhat )

    for i in range(0,3):
      del2_Uhat[i] = -grid.ksqr*Uhat[i] + diff_y2(Uhat[i])
    RHS_implicit[0] = main.nu*del2_Uhat[0] - 1j*grid.k1*main.phat
    RHS_implicit[1] = main.nu*del2_Uhat[1] - diff_y(main.phat)
    RHS_implicit[2] = main.nu*del2_Uhat[2] - 1j*grid.k3*main.phat
    return RHS_explicit,RHS_implicit
  
  def evalQL(Uhat,grid,myFFT,main,filter_mat):
    UhatF = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    for i in range(0,3):
      UhatF[i] = filter_mat*Uhat[i] 
    RHS_explicit,RHS_implicit = evalRHS(Uhat,grid,myFFT,main) 
    RHS = RHS_explicit #+ RHS_implicit
    RHS_explicit,RHS_implicit = evalRHS(UhatF,grid,myFFT,main) 
    RHS_F = RHS_explicit #+ RHS_implicit
    F = RHS - RHS_F
    return F

  main.uhat = grid.dealias_2x*main.uhat
  main.vhat = grid.dealias_2x*main.vhat
  main.what = grid.dealias_2x*main.what
  main.phat = grid.dealias_2x*main.phat

  Uhat = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
  Uhat[0] = main.uhat[:,:,:]
  Uhat[1] = main.vhat[:,:,:]
  Uhat[2] = main.what[:,:,:]
  Uhat_f = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
  Uhat_f[0] = grid.test_filter*main.uhat[:,:,:]
  Uhat_f[1] = grid.test_filter*main.vhat[:,:,:]
  Uhat_f[2] = grid.test_filter*main.what[:,:,:]
  main.RHS_explicit,main.RHS_implicit = evalRHS(Uhat,grid,myFFT,main)
  RHS = main.RHS_explicit# + main.RHS_implicit
  RHSf_explicit,RHSf_implicit = evalRHS(Uhat_f,grid,myFFT,main)
  RHS_f = RHSf_explicit #+ RHSf_implicit

  RHSnorm = np.linalg.norm(RHS)
  eps = 1.e-5
  PLQLU   = evalQL(Uhat   + eps*RHS,grid,myFFT,main,grid.dealias_2x )/eps
  PLQLU_f = evalQL(Uhat_f + eps*RHS_f,grid,myFFT,main,grid.test_filter)/eps
  #print(np.linalg.norm(PLQLU),np.linalg.norm(PLQLU_f))
  LeonardStress = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
  for i in range(0,3):
    LeonardStress[i] = grid.dealias_2x*(RHS_f[i] - RHS[i])
  #print(np.linalg.norm(LeonardStress),np.linalg.norm(PLQLU),np.linalg.norm(PLQLU_f))

  ## Now compute energy up to test filter
  LE = 0. + 0j
  PLQLUE = 0. + 0j
  PLQLU_fE = 0. + 0j

  LE = np.zeros((3,grid.N2),dtype='complex')
  PLQLUE = np.zeros((3,grid.N2),dtype='complex')
  PLQLU_fE = np.zeros((3,grid.N2),dtype='complex')

  for i in range(0,3):
    LE[i] = LE[i] +  np.sum(LeonardStress[i,:,:,1:grid.N3/2]*np.conj(Uhat_f[i,:,:,1:grid.N3/2]*2) ,axis=(0,2) ) + \
             np.sum(LeonardStress[i,:,:,0]*np.conj(Uhat_f[i,:,:,0]) ,axis=(0))

    PLQLUE[i] = PLQLUE[i] +  np.sum(PLQLU[i,:,:,1:grid.N3/2]*np.conj(Uhat_f[i,:,:,1:grid.N3/2]*2) ,axis=(0,2) ) + \
             np.sum(PLQLU[i,:,:,0]*np.conj(Uhat_f[i,:,:,0]),axis=(0))

    PLQLU_fE[i] = PLQLU_fE[i] +  np.sum(PLQLU_f[i,:,:,1:grid.N3/2]*np.conj(Uhat_f[i,:,:,1:grid.N3/2]*2) ,axis=(0,2)) + \
             np.sum(PLQLU_f[i,:,:,0]*np.conj(Uhat_f[i,:,:,0]),axis=(0))
 
  comm = MPI.COMM_WORLD
  num_processes = comm.Get_size()
  mpi_rank = comm.Get_rank()
  LE_loc = comm.gather(LE,root = 0)
  if (mpi_rank == 0):
    LE_total = np.zeros((3,grid.N2),dtype='complex')
    for j in range(0,num_processes):
      LE_total[:] += LE_loc[j]
    LE_total[:] = LE_total[:] / num_processes
    for j in range(1,num_processes):
      comm.send(LE_total, dest=j)
  else:
    LE_total = comm.recv(source=0)

  PLQLUE_loc = comm.gather(PLQLUE,root = 0)
  if (mpi_rank == 0):
    PLQLUE_total = np.zeros((3,grid.N2),dtype='complex')
    for j in range(0,num_processes):
      PLQLUE_total[:] += PLQLUE_loc[j]
    PLQLUE_total = PLQLUE_total / num_processes
    for j in range(1,num_processes):
      comm.send(PLQLUE_total, dest=j)
  else:
    PLQLUE_total = comm.recv(source=0)

  PLQLUfE_loc = comm.gather(PLQLU_fE,root = 0)
  if (mpi_rank == 0):
    PLQLUfE_total = np.zeros((3,grid.N2),dtype='complex')
    for j in range(0,num_processes):
      PLQLUfE_total[:] += PLQLUfE_loc[j]
    PLQLUfE_total = PLQLUfE_total / num_processes
    for j in range(1,num_processes):
      comm.send(PLQLUfE_total, dest=j)
  else:
    PLQLUfE_total = comm.recv(source=0)
  #tau =   np.real( LE_total ) / (np.real(PLQLUE_total)  - 1.5*np.real(PLQLUfE_total) + 1.e-100  )
  LE_total = np.sum(LE_total,axis=0)
  PLQLUE_total = np.sum(PLQLUE_total,axis=0)
  PLQLUfE_total = np.sum(PLQLUfE_total,axis=0)

  tau =   np.real( LE_total ) / (np.real(PLQLUE_total)  - 1.5*np.real(PLQLUfE_total) + 1.e-100  )
  main.tau  = tau
  #if (mpi_rank == 0):
        #print(tau0)
        #print(main.tau)
    #print(abs(LE_total))
    #print(abs(PLQLUE_total))
    #print(abs(PLQLUfE_total))

  main.w0_u[:,:,:,0] = grid.dealias_2x*main.tau[None,:,None]*PLQLU[0]
  main.w0_v[:,:,:,0] = grid.dealias_2x*main.tau[None,:,None]*PLQLU[1]
  main.w0_w[:,:,:,0] = grid.dealias_2x*main.tau[None,:,None]*PLQLU[2]
  for i in range(0,3):
    main.RHS_explicit[i] = grid.dealias_2x*main.RHS_explicit[i]
    main.RHS_implicit[i] = grid.dealias_2x*main.RHS_implicit[i]
 #
  main.RHS_explicit[0] += main.w0_u[:,:,:,0] 
  main.RHS_explicit[1] += main.w0_v[:,:,:,0] 
  main.RHS_explicit[2] += main.w0_w[:,:,:,0] 



def getRHS_vort_dtau_2(main,grid,myFFT):
  def evalRHS(Uhat,grid,myFFT,main):
    U = np.zeros((3,grid.N1,grid.Npy,grid.N3))
    omega = np.zeros((3,grid.N1,grid.Npy,grid.N3))
    omegahat = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    U[0] = myFFT.myifft3D(Uhat[0])
    U[1] = myFFT.myifft3D(Uhat[1])
    U[2] = myFFT.myifft3D(Uhat[2])
  
    ## compute vorticity
    omegahat[0] = diff_y(Uhat[2]) - 1j*grid.k3*Uhat[1]
    omegahat[1] = 1j*grid.k3*Uhat[0] - 1j*grid.k1*Uhat[2]
    omegahat[2] = 1j*grid.k1*Uhat[1] - diff_y(Uhat[0])
 
    omega[0] = myFFT.myifft3D(omegahat[0])
    omega[1] = myFFT.myifft3D(omegahat[1])
    omega[2] = myFFT.myifft3D(omegahat[2])
  
    uu = U[0]*U[0]
    vv = U[1]*U[1]
    ww = U[2]*U[2]
  
    vom3 = U[1]*omega[2]
    wom2 = U[2]*omega[1]
    uom3 = U[0]*omega[2]
    wom1 = U[2]*omega[0]
    uom2 = U[0]*omega[1]
    vom1 = U[1]*omega[0]
  
  
    uuhat = myFFT.myfft3D(uu) 
    vvhat = myFFT.myfft3D(vv) 
    wwhat = myFFT.myfft3D(ww) 
  
    vom3_hat = myFFT.myfft3D(vom3) 
    wom2_hat = myFFT.myfft3D(wom2) 
    uom3_hat = myFFT.myfft3D(uom3) 
    wom1_hat = myFFT.myfft3D(wom1) 
    uom2_hat = myFFT.myfft3D(uom2) 
    vom1_hat = myFFT.myfft3D(vom1) 
  
    vsqrhat = 0.5*( uuhat + vvhat + wwhat)

    RHS_explicit = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    RHS_implicit = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    del2_Uhat = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')

    RHS_explicit[0] = -( wom2_hat -vom3_hat + 1j*grid.k1*vsqrhat ) - main.dP ### mean pressure gradient only
    RHS_explicit[1] = -( uom3_hat -wom1_hat + diff_y(vsqrhat)    )
    RHS_explicit[2] = -( vom1_hat -uom2_hat + 1j*grid.k3*vsqrhat )

    for i in range(0,3):
      del2_Uhat[i] = -grid.ksqr*Uhat[i] + diff_y2(Uhat[i])
    RHS_implicit[0] = main.nu*del2_Uhat[0] - 1j*grid.k1*main.phat
    RHS_implicit[1] = main.nu*del2_Uhat[1] - diff_y(main.phat)
    RHS_implicit[2] = main.nu*del2_Uhat[2] - 1j*grid.k3*main.phat
    return RHS_explicit,RHS_implicit
  
  def evalQL(Uhat,grid,myFFT,main,filter_mat):
    UhatF = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    for i in range(0,3):
      UhatF[i] = filter_mat*Uhat[i] 
    RHS_explicit,RHS_implicit = evalRHS(Uhat,grid,myFFT,main) 
    RHS = RHS_explicit #+ RHS_implicit
    RHS_explicit,RHS_implicit = evalRHS(UhatF,grid,myFFT,main) 
    RHS_F = RHS_explicit #+ RHS_implicit
    F = RHS - RHS_F
    return F

  main.uhat = grid.dealias_2x*main.uhat
  main.vhat = grid.dealias_2x*main.vhat
  main.what = grid.dealias_2x*main.what
  main.phat = grid.dealias_2x*main.phat

  Uhat = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
  Uhat[0] = main.uhat[:,:,:]
  Uhat[1] = main.vhat[:,:,:]
  Uhat[2] = main.what[:,:,:]
  Uhat_f = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
  Uhat_f[0] = grid.test_filter*main.uhat[:,:,:]
  Uhat_f[1] = grid.test_filter*main.vhat[:,:,:]
  Uhat_f[2] = grid.test_filter*main.what[:,:,:]
  main.RHS_explicit,main.RHS_implicit = evalRHS(Uhat,grid,myFFT,main)
  RHS = main.RHS_explicit# + main.RHS_implicit
  RHSf_explicit,RHSf_implicit = evalRHS(Uhat_f,grid,myFFT,main)
  RHS_f = RHSf_explicit #+ RHSf_implicit

  RHSnorm = np.linalg.norm(RHS)
  eps = 1.e-5
  PLQLU   = evalQL(Uhat   + eps*RHS,grid,myFFT,main,grid.dealias_2x )/eps
  PLQLU_f = evalQL(Uhat_f + eps*RHS_f,grid,myFFT,main,grid.test_filter)/eps
  #print(np.linalg.norm(PLQLU),np.linalg.norm(PLQLU_f))
  LeonardStress = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
  for i in range(0,3):
    LeonardStress[i] = grid.dealias_2x*(RHS_f[i] - RHS[i])
  #print(np.linalg.norm(LeonardStress),np.linalg.norm(PLQLU),np.linalg.norm(PLQLU_f))

  ## Now compute energy up to test filter
  LE = 0. + 0j
  PLQLUE = 0. + 0j
  PLQLU_fE = 0. + 0j

  LE = np.zeros(3,dtype='complex')
  PLQLUE = np.zeros(3,dtype='complex')
  PLQLU_fE = np.zeros(3,dtype='complex')

  for i in range(0,3):
    LE[i] = LE[i] +  np.sum(LeonardStress[i,:,:,1:grid.N3/2]*np.conj(Uhat_f[i,:,:,1:grid.N3/2]*2) ) + \
             np.sum(LeonardStress[i,:,:,0]*np.conj(Uhat_f[i,:,:,0]))

    PLQLUE[i] = PLQLUE[i] +  np.sum(PLQLU[i,:,:,1:grid.N3/2]*np.conj(Uhat_f[i,:,:,1:grid.N3/2]*2) ) + \
             np.sum(PLQLU[i,:,:,0]*np.conj(Uhat_f[i,:,:,0]))

    PLQLU_fE[i] = PLQLU_fE[i] +  np.sum(PLQLU_f[i,:,:,1:grid.N3/2]*np.conj(Uhat_f[i,:,:,1:grid.N3/2]*2) ) + \
             np.sum(PLQLU_f[i,:,:,0]*np.conj(Uhat_f[i,:,:,0]))
 
  comm = MPI.COMM_WORLD
  num_processes = comm.Get_size()
  mpi_rank = comm.Get_rank()
  LE_loc = comm.gather(LE,root = 0)
  if (mpi_rank == 0):
    LE_total = np.zeros(3,dtype='complex')
    for j in range(0,num_processes):
      LE_total[:] += LE_loc[j]
    LE_total[:] = LE_total[:] / num_processes
    for j in range(1,num_processes):
      comm.send(LE_total, dest=j)
  else:
    LE_total = comm.recv(source=0)

  PLQLUE_loc = comm.gather(PLQLUE,root = 0)
  if (mpi_rank == 0):
    PLQLUE_total = np.zeros(3,dtype='complex')
    for j in range(0,num_processes):
      PLQLUE_total[:] += PLQLUE_loc[j]
    PLQLUE_total = PLQLUE_total / num_processes
    for j in range(1,num_processes):
      comm.send(PLQLUE_total, dest=j)
  else:
    PLQLUE_total = comm.recv(source=0)

  PLQLUfE_loc = comm.gather(PLQLU_fE,root = 0)
  if (mpi_rank == 0):
    PLQLUfE_total = np.zeros(3,dtype='complex')
    for j in range(0,num_processes):
      PLQLUfE_total[:] += PLQLUfE_loc[j]
    PLQLUfE_total = PLQLUfE_total / num_processes
    for j in range(1,num_processes):
      comm.send(PLQLUfE_total, dest=j)
  else:
    PLQLUfE_total = comm.recv(source=0)
  #tau =   np.real( LE_total ) / (np.real(PLQLUE_total)  - 1.5*np.real(PLQLUfE_total) + 1.e-100  )
  LE_total = np.sum(LE_total)
  PLQLUE_total = np.sum(PLQLUE_total)
  PLQLUfE_total = np.sum(PLQLUfE_total)

  #print((grid.kcx/grid.test_kcx)**1.5,grid.kcx,grid.test_kcx)
  tau =   np.real( LE_total ) / (np.real(PLQLUE_total)  - (grid.kcx/grid.test_kcx)**1.5*np.real(PLQLUfE_total) + 1.e-100  )
  main.tau  = tau
  if (mpi_rank == 0):
        #print(tau0)
        print(main.tau)
    #print(abs(LE_total))
    #print(abs(PLQLUE_total))
    #print(abs(PLQLUfE_total))

  main.w0_u[:,:,:,0] = grid.dealias_2x*main.tau*PLQLU[0]
  main.w0_v[:,:,:,0] = grid.dealias_2x*main.tau*PLQLU[1]
  main.w0_w[:,:,:,0] = grid.dealias_2x*main.tau*PLQLU[2]
  for i in range(0,3):
    main.RHS_explicit[i] = grid.dealias_2x*main.RHS_explicit[i]
    main.RHS_implicit[i] = grid.dealias_2x*main.RHS_implicit[i]
 #
  main.RHS_explicit[0] += main.w0_u[:,:,:,0] 
  main.RHS_explicit[1] += main.w0_v[:,:,:,0] 
  main.RHS_explicit[2] += main.w0_w[:,:,:,0] 



def getRHS_vort_stau_2(main,grid,myFFT):
  def evalRHS(Uhat,grid,myFFT,main):
    U = np.zeros((3,grid.N1,grid.Npy,grid.N3))
    omega = np.zeros((3,grid.N1,grid.Npy,grid.N3))
    omegahat = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    U[0] = myFFT.myifft3D(Uhat[0])
    U[1] = myFFT.myifft3D(Uhat[1])
    U[2] = myFFT.myifft3D(Uhat[2])
  
    ## compute vorticity
    omegahat[0] = diff_y(Uhat[2]) - 1j*grid.k3*Uhat[1]
    omegahat[1] = 1j*grid.k3*Uhat[0] - 1j*grid.k1*Uhat[2]
    omegahat[2] = 1j*grid.k1*Uhat[1] - diff_y(Uhat[0])
 
    omega[0] = myFFT.myifft3D(omegahat[0])
    omega[1] = myFFT.myifft3D(omegahat[1])
    omega[2] = myFFT.myifft3D(omegahat[2])
  
    uu = U[0]*U[0]
    vv = U[1]*U[1]
    ww = U[2]*U[2]
  
    vom3 = U[1]*omega[2]
    wom2 = U[2]*omega[1]
    uom3 = U[0]*omega[2]
    wom1 = U[2]*omega[0]
    uom2 = U[0]*omega[1]
    vom1 = U[1]*omega[0]
  
  
    uuhat = myFFT.myfft3D(uu) 
    vvhat = myFFT.myfft3D(vv) 
    wwhat = myFFT.myfft3D(ww) 
  
    vom3_hat = myFFT.myfft3D(vom3) 
    wom2_hat = myFFT.myfft3D(wom2) 
    uom3_hat = myFFT.myfft3D(uom3) 
    wom1_hat = myFFT.myfft3D(wom1) 
    uom2_hat = myFFT.myfft3D(uom2) 
    vom1_hat = myFFT.myfft3D(vom1) 
  
    vsqrhat = 0.5*( uuhat + vvhat + wwhat)

    RHS_explicit = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    RHS_implicit = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    del2_Uhat = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')

    RHS_explicit[0] = -( wom2_hat -vom3_hat + 1j*grid.k1*vsqrhat ) - main.dP ### mean pressure gradient only
    RHS_explicit[1] = -( uom3_hat -wom1_hat + diff_y(vsqrhat)    )
    RHS_explicit[2] = -( vom1_hat -uom2_hat + 1j*grid.k3*vsqrhat )

    for i in range(0,3):
      del2_Uhat[i] = -grid.ksqr*Uhat[i] + diff_y2(Uhat[i])
    RHS_implicit[0] = main.nu*del2_Uhat[0] - 1j*grid.k1*main.phat
    RHS_implicit[1] = main.nu*del2_Uhat[1] - diff_y(main.phat)
    RHS_implicit[2] = main.nu*del2_Uhat[2] - 1j*grid.k3*main.phat
    return RHS_explicit,RHS_implicit
  
  def evalQL(Uhat,grid,myFFT,main):
    UhatF = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    UhatG = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    for i in range(0,3):
      UhatF[i] = grid.dealias_2x*Uhat[i] 
    RHS_explicit,RHS_implicit = evalRHS(Uhat,grid,myFFT,main) 
    RHS = RHS_explicit + RHS_implicit
    RHS_explicit,RHS_implicit = evalRHS(UhatF,grid,myFFT,main) 
    RHS_F = RHS_explicit + RHS_implicit
    F = RHS - RHS_F
    return F

  main.uhat = grid.dealias_2x*main.uhat
  main.vhat = grid.dealias_2x*main.vhat
  main.what = grid.dealias_2x*main.what
  main.phat = grid.dealias_2x*main.phat

  Uhat = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
  Uhat[0] = main.uhat[:,:,:]
  Uhat[1] = main.vhat[:,:,:]
  Uhat[2] = main.what[:,:,:]

  main.RHS_explicit,main.RHS_implicit = evalRHS(Uhat,grid,myFFT,main)
  RHS = main.RHS_explicit + main.RHS_implicit
  #RHSnorm = np.linalg.norm(RHS)
  eps = 1.e-5
  PLQLU = evalQL(Uhat + eps*RHS,grid,myFFT,main)/eps
  main.w0_u = grid.dealias_2x*main.tau0*PLQLU[0]
  main.w0_v = grid.dealias_2x*main.tau0*PLQLU[1]
  main.w0_w = grid.dealias_2x*main.tau0*PLQLU[2]
  for i in range(0,3):
    main.RHS_explicit[i] = grid.dealias_2x*main.RHS_explicit[i]
    main.RHS_implicit[i] = grid.dealias_2x*main.RHS_implicit[i]
 #
  main.RHS_explicit[0] += main.w0_u 
  main.RHS_explicit[1] += main.w0_v 
  main.RHS_explicit[2] += main.w0_w 

def lineSolve2(main,grid,myFFT,i,I,I2):
  N2 = grid.N2
  ms = np.linspace(0,grid.N2*2/3-1,grid.N2*2/3)
  cm = np.ones(grid.N2*2/3)
  cm[0] = 2.
  altarray = (-np.ones(grid.N2*2/3))**(np.linspace(0,(grid.N2*2/3)-1,grid.N2*2/3))
  for k in range(0,grid.N3/2+1):
    if (grid.k1[i,0,k] == 0 and grid.k3[i,0,k] == 0):
      ### By continuity vhat = 0, don't need to solve for it.
      ### also by definition, p is the fixed pressure gradient, don't need to solve for it. 
      ### Just need to find u and w
      RHS = np.zeros((grid.N2*2/3),dtype='complex')
      RHSu = np.zeros((grid.N2*2/3),dtype='complex')
      RHSu[:] = main.uhat[i,0:N2*2/3,k] + main.dt/2.*(3.*main.RHS_explicit[0,i,0:N2*2/3,k] - main.RHS_explicit_old[0,i,0:N2*2/3,k]) + \
                main.dt/2.*( main.RHS_implicit[0,i,0:N2*2/3,k] )
      RHSw = np.zeros((grid.N2*2/3),dtype='complex')
      RHSw[:] = main.what[i,0:N2*2/3,k] + main.dt/2.*(3.*main.RHS_explicit[2,i,0:N2*2/3,k] - main.RHS_explicit_old[2,i,0:N2*2/3,k]) + \
                main.dt/2.*( main.RHS_implicit[2,i,0:N2*2/3,k] )

      ## Now create entire LHS matrix
      LHSMAT = np.zeros((grid.N2*2/3,grid.N2*2/3),dtype='complex')
      ## insert in the linear contribution for the momentum equations
      LHSMAT[0,:] = 1.
      LHSMAT[1,:] = altarray
      np.fill_diagonal(LHSMAT[2::,2::],-main.nu*main.dt*ms[2::] -1./(2.*ms[2::]-2.) - 1./(2.*ms[2::]+2.) )
      np.fill_diagonal(LHSMAT[2::,0:-2],cm[0:-2]/(2.*ms[2::] - 2.) )
      np.fill_diagonal(LHSMAT[2::,4::],1./(2.*ms[2::]+2.) )
      RHSu = np.append(RHSu,np.zeros(2))
      RHSw = np.append(RHSw,np.zeros(2))

      RHS[2::] = (cm[0:-2]*RHSu[0:-4]  - RHSu[2:-2]) / (2.*ms[2::] - 2.)  - (RHSu[2:-2] - RHSu[4::] )/(2.*ms[2::] + 2.)
      main.uhat[i,0:N2*2/3,k] = np.linalg.solve(LHSMAT,RHS)
      RHS[2::] = (cm[0:-2]*RHSw[0:-4]  - RHSw[2:-2]) / (2.*ms[2::] - 2.)  - (RHSw[2:-2] - RHSw[4::] )/(2.*ms[2::] + 2.)
      main.what[i,0:N2*2/3,k] = np.linalg.solve(LHSMAT,RHS)
      main.vhat[i,0:N2*2/3,k] = 0.
    else:
      if (abs(grid.k3[i,0,k]) <= grid.kcz): # don't bother solving for dealiased wave numbers
        t0 = time.time()
        ## SOLUTION VECTOR LOOKS LIKE
        #[ u0,v0,w0,ph0,u1,v1,w1,ph1,...,un,vn,wn]
        ## Form linear matrix for Crank Nicolson terms
        ## Now create RHS solution vector
        RHS =  np.zeros((N2*2/3)*4,dtype='complex')
        RHSu = np.zeros(N2*2/3,dtype='complex')
        RHSv = np.zeros(N2*2/3,dtype='complex')
        RHSw = np.zeros(N2*2/3,dtype='complex')

        RHSu = main.uhat[i,0:N2*2/3,k] +  main.dt/2.*(3.*main.RHS_explicit[0,i,0:N2*2/3,k] - main.RHS_explicit_old[0,i,0:N2*2/3,k]) + main.dt/2.*main.RHS_implicit[0,i,0:N2*2/3,k]
        RHSv = main.vhat[i,0:N2*2/3,k] +  main.dt/2.*(3.*main.RHS_explicit[1,i,0:N2*2/3,k] - main.RHS_explicit_old[1,i,0:N2*2/3,k]) + main.dt/2.*main.RHS_implicit[1,i,0:N2*2/3,k]
        RHSw = main.what[i,0:N2*2/3,k] +  main.dt/2.*(3.*main.RHS_explicit[2,i,0:N2*2/3,k] - main.RHS_explicit_old[2,i,0:N2*2/3,k]) + main.dt/2.*main.RHS_implicit[2,i,0:N2*2/3,k]

        LHSMATu = np.zeros((grid.N2*2/3,grid.N2*2/3),dtype='complex')
        LHSMATu[0,:] = 1.
        LHSMATu[1,:] = altarray
        np.fill_diagonal(LHSMATu[2::,2::],-main.nu*main.dt*ms[2::] \
                                       -(1.+main.nu*main.dt*grid.ksqr[i,0,k]*0.5)/(2.*ms[2::]-2.) \
                                       -(1.+main.nu*main.dt*grid.ksqr[i,0,k]*0.5)/(2.*ms[2::]+2.) )
        np.fill_diagonal(LHSMATu[2::,0:-2],cm[0:-2]/(2.*ms[2::]-2.)*(1. + main.nu*main.dt*grid.ksqr[i,0,k]*0.5) )
        np.fill_diagonal(LHSMATu[2::,4::],       1./(2.*ms[2::]+2.)*(1. + main.nu*main.dt*grid.ksqr[i,0,k]*0.5) )
        LHSMATp = np.zeros((grid.N2*2/3,grid.N2*2/3),dtype='complex')
        np.fill_diagonal(LHSMATp[2::,0:-2],main.dt*0.5*1j*cm[0:-2]/(2.*ms[2::]-2.) )
        np.fill_diagonal(LHSMATp[2::, 2::],main.dt*0.5*1j*( -1./(2.*ms[2::]-2.) - 1./(2.*ms[2::]+2.) ) )
        np.fill_diagonal(LHSMATp[2::, 4::],main.dt*0.5*1j*1./(2.*ms[2::]+2.) )
        LHSMATpv = np.zeros((grid.N2*2/3,grid.N2*2/3),dtype='complex')
        np.fill_diagonal(LHSMATpv[2::,1::],main.dt*0.5)
        np.fill_diagonal(LHSMATpv[2::,3::],-main.dt*0.5)


        LHSMAT = np.zeros(( (N2*2/3)*4,(N2*2/3)*4),dtype='complex')
        LHSMAT[0::4,0::4] = LHSMATu
        LHSMAT[1::4,1::4] = LHSMATu
        LHSMAT[2::4,2::4] = LHSMATu
        LHSMAT[0::4,3::4] = grid.k1[i,0,k]*LHSMATp
        LHSMAT[1::4,3::4] = LHSMATpv
        LHSMAT[2::4,3::4] = grid.k3[i,0,k]*LHSMATp
        ##========== Continuity Equation ================
        np.fill_diagonal(LHSMAT[3::4,0::4], 1j*grid.k1[i,0,k]*cm[:] )
        np.fill_diagonal(LHSMAT[3::4,8::4],-1j*grid.k1[i,0,k] )
        np.fill_diagonal(LHSMAT[3::4,5::4], 2.*ms[1::] )
        np.fill_diagonal(LHSMAT[3::4,2::4], 1j*grid.k3[i,0,k]*cm[:])
        np.fill_diagonal(LHSMAT[3::4,10::4],-1j*grid.k3[i,0,k])

        RHSu = np.append(RHSu,np.zeros(2))
        RHSv = np.append(RHSv,np.zeros(2))
        RHSw = np.append(RHSw,np.zeros(2))

        RHS[8::4] = (cm[0:-2]*RHSu[0:-4]  - RHSu[2:-2]) / (2.*ms[2::] - 2.)  - \
                            (RHSu[2:-2] - RHSu[4::] )/(2.*ms[2::] + 2.)

        RHS[9::4] = (cm[0:-2]*RHSv[0:-4]  - RHSv[2:-2]) / (2.*ms[2::] - 2.)  - \
                            (RHSv[2:-2] - RHSv[4::] )/(2.*ms[2::] + 2.)

        RHS[10::4] = (cm[0:-2]*RHSw[0:-4]  - RHSw[2:-2]) / (2.*ms[2::] - 2.)  - \
                            (RHSw[2:-2] - RHSw[4::] )/(2.*ms[2::] + 2.)

        RHS[3::4] = 0.

        t1 = time.time()
    #    solver = scipy.sparse.linalg.factorized( scipy.sparse.csc_matrix(LHSMAT))
    #    U = solver(RHS)
    #    U = np.linalg.solve(LHSMAT,RHS)
        U = (scipy.sparse.linalg.spsolve( scipy.sparse.csc_matrix(LHSMAT),RHS, permc_spec="NATURAL") )
    #    U = (scipy.sparse.linalg.bicgstab( scipy.sparse.csc_matrix(LHSMAT),RHS,tol=1e-14) )[0]
        main.uhat[i,0:N2*2/3,k] = U[0::4]#*grid.dealias[0,:,0]
        main.vhat[i,0:N2*2/3,k] = U[1::4]#*grid.dealias[0,:,0]
        main.what[i,0:N2*2/3,k] = U[2::4]#*grid.dealias[0,:,0]
        main.phat[i,0:N2*2/3,k] = U[3::4]#*grid.dealias[0,:,0]
        #print(1j*grid.k1[i,0,k]*main.uhat[i,N2*2/3-2,k] + main.vhat[i,N2*2/3-1,k]*2.*(N2*2/3-1) + 1j*grid.k3[i,0,k]*main.what[i,N2*2/3-2,k] ) 
       # print(np.linalg.norm(dot(LHSMAT,U)[3::4] ) )
       # print(LHSMAT[-4,0::-7])
        main.LHSMAT = LHSMAT
       # main.RHS = RHS



def lineSolve_dealias2x(main,grid,myFFT,i,I,I2):
  N2 = grid.N2
  altarray = (-np.ones(grid.kcy_int))**(np.linspace(0,(grid.kcy_int)-1,grid.kcy_int))
  for k in range(0,grid.N3/2+1):
    if (grid.k1[i,0,k] == 0 and grid.k3[i,0,k] == 0):
      ### By continuity vhat = 0, don't need to solve for it.
      ### also by definition, p is the fixed pressure gradient, don't need to solve for it. 
      ### Just need to find u and w
      F = np.zeros((grid.kcy_int,grid.kcy_int),dtype='complex')
      F[:,:] =  -main.nu*( grid.A2[:,:]  )
      RHSu = np.zeros((grid.kcy_int),dtype='complex')
      RHSu[:] = main.uhat[i,0:grid.kcy_int,k] + main.dt/2.*(3.*main.RHS_explicit[0,i,0:grid.kcy_int,k] - main.RHS_explicit_old[0,i,0:grid.kcy_int,k]) + \
                main.dt/2.*( main.RHS_implicit[0,i,0:grid.kcy_int,k] )
      RHSw = np.zeros((grid.kcy_int),dtype='complex')
      RHSw[:] = main.what[i,0:grid.kcy_int,k] + main.dt/2.*(3.*main.RHS_explicit[2,i,0:grid.kcy_int,k] - main.RHS_explicit_old[2,i,0:grid.kcy_int,k]) + \
                main.dt/2.*( main.RHS_implicit[2,i,0:grid.kcy_int,k] )

      ## Now create entire LHS matrix
      LHSMAT = np.zeros((grid.kcy_int,grid.kcy_int),dtype='complex')
      ## insert in the linear contribution for the momentum equations
      LHSMAT[:,:] = np.eye(grid.kcy_int) + 0.5*main.dt*F[:,:]
      ## Finally setup boundary condtions
      LHSMAT[-2,:] = 1.#*grid.dealias[0,:,0]
      LHSMAT[-1,:] = altarray#*grid.dealias[0,:,0]
      RHSu[-2::] = 0.
      RHSw[-2::] = 0.
      main.uhat[i,0:grid.kcy_int,k] = np.linalg.solve(LHSMAT,RHSu)
      main.what[i,0:grid.kcy_int,k] = np.linalg.solve(LHSMAT,RHSw)
      main.vhat[i,0:grid.kcy_int,k] = 0. 
    else:
      if (abs(grid.k3[i,0,k]) <= grid.kcz): # don't bother solving for dealiased wave numbers
        t0 = time.time()
        ## SOLUTION VECTOR LOOKS LIKE
        #[ u0,v0,w0,ph0,u1,v1,w1,ph1,...,un,vn,wn]
        ## Form linear matrix for Crank Nicolson terms
        F = np.zeros(( (grid.kcy_int)*4-1,(grid.kcy_int)*4-1),dtype='complex')
        #F = scipy.sparse.csc_matrix((grid.N2*4-1, grid.N2*4-1), dtype=complex).toarray()
        F[0::4,0::4] = -main.nu*( grid.A2[:,:] - grid.ksqr[i,0,k]*I2[:,:] )###Viscous terms
        F[1::4,1::4] = F[0::4,0::4]  ### put into v eqn as well
        F[2::4,2::4] = F[0::4,0::4]  ### put into w eqn as well
        np.fill_diagonal( F[0::4,3::4],1j*grid.k1[i,0,k] )  ## now add pressure to u eqn
        F[1:-2:4,3::4] = grid.A1p[:,:]                      ## v eqn
        np.fill_diagonal( F[2::4,3::4],1j*grid.k3[i,0,k] )  ## w eqn
   
        ## Now create RHS solution vector
        RHS = np.zeros(( (grid.kcy_int)*4-1),dtype='complex')
        RHS[0::4] = main.uhat[i,0:grid.kcy_int,k] +  main.dt/2.*(3.*main.RHS_explicit[0,i,0:grid.kcy_int,k] - main.RHS_explicit_old[0,i,0:grid.kcy_int,k]) + main.dt/2.*main.RHS_implicit[0,i,0:grid.kcy_int,k]
        RHS[1::4] = main.vhat[i,0:grid.kcy_int,k] +  main.dt/2.*(3.*main.RHS_explicit[1,i,0:grid.kcy_int,k] - main.RHS_explicit_old[1,i,0:grid.kcy_int,k]) + main.dt/2.*main.RHS_implicit[1,i,0:grid.kcy_int,k]
        RHS[2::4] = main.what[i,0:grid.kcy_int,k] +  main.dt/2.*(3.*main.RHS_explicit[2,i,0:grid.kcy_int,k] - main.RHS_explicit_old[2,i,0:grid.kcy_int,k]) + main.dt/2.*main.RHS_implicit[2,i,0:grid.kcy_int,k]
  
        LHSMAT = np.zeros(( (grid.kcy_int)*4-1,(grid.kcy_int)*4-1),dtype='complex')
        #LHSMAT = scipy.sparse.csc_matrix((grid.N2*4-1, grid.N2*4-1), dtype=complex).toarray()
        ## insert in the linear contribution for the momentum equations
        LHSMAT[:,:] = I + 0.5*main.dt*F[:,:]
        ## Now add the continuity equation (evaluated at half grid points)
        LHSMAT[3::4,:] = 0.
        np.fill_diagonal( LHSMAT[3::4,0::4] , 1j*grid.k1[i,0,k] )  ##  du/dx
        LHSMAT[3::4,1::4] = grid.A1[0:-1,:]                        ##  dv/dy
        np.fill_diagonal( LHSMAT[3::4,2::4] , 1j*grid.k3[i,0,k] )  ##  dw/dz
        ## Finally setup boundary condtions
        RHS[-7::] = 0.
        LHSMAT[-7,:] = 0.
        LHSMAT[-6,:] = 0.
        LHSMAT[-5,:] = 0.
        LHSMAT[-3,:] = 0.
        LHSMAT[-2,:] = 0.
        LHSMAT[-1,:] = 0.
    
        LHSMAT[-7,0::4] = 1. / (grid.N1 * grid.N3) #* grid.dealias[0,0,0]
        LHSMAT[-6,1::4] = 1. / (grid.N1 * grid.N3) #* grid.dealias[0,:,0]
        LHSMAT[-5,2::4] = 1. / (grid.N1 * grid.N3) #* grid.dealias[0,:,0]
        LHSMAT[-3,0::4] = altarray[0:grid.kcy_int] #* grid.dealias[0,:,0]
        LHSMAT[-2,1::4] = altarray[0:grid.kcy_int] #* grid.dealias[0,:,0]
        LHSMAT[-1,2::4] = altarray[0:grid.kcy_int] #* grid.dealias[0,:,0]

  
        t1 = time.time() 
    #    solver = scipy.sparse.linalg.factorized( scipy.sparse.csc_matrix(LHSMAT))
    #    U = solver(RHS)
    #    U = np.linalg.solve(LHSMAT,RHS)
        U = (scipy.sparse.linalg.spsolve( scipy.sparse.csc_matrix(LHSMAT),RHS, permc_spec="NATURAL") )
    #    U = (scipy.sparse.linalg.bicgstab( scipy.sparse.csc_matrix(LHSMAT),RHS,tol=1e-14) )[0]
        main.uhat[i,0:grid.kcy_int,k] = U[0::4]#*grid.dealias[0,:,0]
        main.vhat[i,0:grid.kcy_int,k] = U[1::4]#*grid.dealias[0,:,0]
        main.what[i,0:grid.kcy_int,k] = U[2::4]#*grid.dealias[0,:,0]
        main.phat[i,0:grid.kcy_int-1,k] = U[3::4]#*grid.dealias[0,:,0]
        main.LHSMAT = LHSMAT
        main.RHS = RHS


def lineSolve(main,grid,myFFT,i,I,I2):
  N2 = grid.N2
  altarray = (-np.ones(grid.N2*2/3))**(np.linspace(0,(grid.N2*2/3)-1,grid.N2*2/3))
  for k in range(0,grid.N3/2+1):
    if (grid.k1[i,0,k] == 0 and grid.k3[i,0,k] == 0):
      ### By continuity vhat = 0, don't need to solve for it.
      ### also by definition, p is the fixed pressure gradient, don't need to solve for it. 
      ### Just need to find u and w
      F = np.zeros((grid.N2*2/3,grid.N2*2/3),dtype='complex')
      F[:,:] =  -main.nu*( grid.A2[:,:]  )
      RHSu = np.zeros((grid.N2*2/3),dtype='complex')
      RHSu[:] = main.uhat[i,0:N2*2/3,k] + main.dt/2.*(3.*main.RHS_explicit[0,i,0:N2*2/3,k] - main.RHS_explicit_old[0,i,0:N2*2/3,k]) + \
                main.dt/2.*( main.RHS_implicit[0,i,0:N2*2/3,k] )
      RHSw = np.zeros((grid.N2*2/3),dtype='complex')
      RHSw[:] = main.what[i,0:N2*2/3,k] + main.dt/2.*(3.*main.RHS_explicit[2,i,0:N2*2/3,k] - main.RHS_explicit_old[2,i,0:N2*2/3,k]) + \
                main.dt/2.*( main.RHS_implicit[2,i,0:N2*2/3,k] )

      ## Now create entire LHS matrix
      LHSMAT = np.zeros((grid.N2*2/3,grid.N2*2/3),dtype='complex')
      ## insert in the linear contribution for the momentum equations
      LHSMAT[:,:] = np.eye(grid.N2*2/3) + 0.5*main.dt*F[:,:]
      ## Finally setup boundary condtions
      LHSMAT[-2,:] = 1.#*grid.dealias[0,:,0]
      LHSMAT[-1,:] = altarray#*grid.dealias[0,:,0]
      RHSu[-2::] = 0.
      RHSw[-2::] = 0.
      main.uhat[i,0:N2*2/3,k] = np.linalg.solve(LHSMAT,RHSu)
      main.what[i,0:N2*2/3,k] = np.linalg.solve(LHSMAT,RHSw)
      main.vhat[i,0:N2*2/3,k] = 0. 
    else:
      if (abs(grid.k3[i,0,k]) <= grid.kcz): # don't bother solving for dealiased wave numbers
        t0 = time.time()
        ## SOLUTION VECTOR LOOKS LIKE
        #[ u0,v0,w0,ph0,u1,v1,w1,ph1,...,un,vn,wn]
        ## Form linear matrix for Crank Nicolson terms
        F = np.zeros(( (N2*2/3)*4-1,(N2*2/3)*4-1),dtype='complex')
        #F = scipy.sparse.csc_matrix((grid.N2*4-1, grid.N2*4-1), dtype=complex).toarray()
        F[0::4,0::4] = -main.nu*( grid.A2[:,:] - grid.ksqr[i,0,k]*I2[:,:] )###Viscous terms
        F[1::4,1::4] = F[0::4,0::4]  ### put into v eqn as well
        F[2::4,2::4] = F[0::4,0::4]  ### put into w eqn as well
        np.fill_diagonal( F[0::4,3::4],1j*grid.k1[i,0,k] )  ## now add pressure to u eqn
        F[1:-2:4,3::4] = grid.A1p[:,:]                      ## v eqn
        np.fill_diagonal( F[2::4,3::4],1j*grid.k3[i,0,k] )  ## w eqn
   
        ## Now create RHS solution vector
        RHS = np.zeros(( (N2*2/3)*4-1),dtype='complex')
        RHS[0::4] = main.uhat[i,0:N2*2/3,k] +  main.dt/2.*(3.*main.RHS_explicit[0,i,0:N2*2/3,k] - main.RHS_explicit_old[0,i,0:N2*2/3,k]) + main.dt/2.*main.RHS_implicit[0,i,0:N2*2/3,k]
        RHS[1::4] = main.vhat[i,0:N2*2/3,k] +  main.dt/2.*(3.*main.RHS_explicit[1,i,0:N2*2/3,k] - main.RHS_explicit_old[1,i,0:N2*2/3,k]) + main.dt/2.*main.RHS_implicit[1,i,0:N2*2/3,k]
        RHS[2::4] = main.what[i,0:N2*2/3,k] +  main.dt/2.*(3.*main.RHS_explicit[2,i,0:N2*2/3,k] - main.RHS_explicit_old[2,i,0:N2*2/3,k]) + main.dt/2.*main.RHS_implicit[2,i,0:N2*2/3,k]
  
        LHSMAT = np.zeros(( (N2*2/3)*4-1,(N2*2/3)*4-1),dtype='complex')
        #LHSMAT = scipy.sparse.csc_matrix((grid.N2*4-1, grid.N2*4-1), dtype=complex).toarray()
        ## insert in the linear contribution for the momentum equations
        LHSMAT[:,:] = I + 0.5*main.dt*F[:,:]
        ## Now add the continuity equation (evaluated at half grid points)
        LHSMAT[3::4,:] = 0.
        np.fill_diagonal( LHSMAT[3::4,0::4] , 1j*grid.k1[i,0,k] )  ##  du/dx
        LHSMAT[3::4,1::4] = grid.A1[0:-1,:]                        ##  dv/dy
        np.fill_diagonal( LHSMAT[3::4,2::4] , 1j*grid.k3[i,0,k] )  ##  dw/dz
        ## Finally setup boundary condtions
        RHS[-7::] = 0.
        LHSMAT[-7,:] = 0.
        LHSMAT[-6,:] = 0.
        LHSMAT[-5,:] = 0.
        LHSMAT[-3,:] = 0.
        LHSMAT[-2,:] = 0.
        LHSMAT[-1,:] = 0.
    
        LHSMAT[-7,0::4] = 1. / (grid.N1 * grid.N3) #* grid.dealias[0,0,0]
        LHSMAT[-6,1::4] = 1. / (grid.N1 * grid.N3) #* grid.dealias[0,:,0]
        LHSMAT[-5,2::4] = 1. / (grid.N1 * grid.N3) #* grid.dealias[0,:,0]
        LHSMAT[-3,0::4] = altarray[0:N2*2/3] #* grid.dealias[0,:,0]
        LHSMAT[-2,1::4] = altarray[0:N2*2/3] #* grid.dealias[0,:,0]
        LHSMAT[-1,2::4] = altarray[0:N2*2/3] #* grid.dealias[0,:,0]

  
        t1 = time.time() 
    #    solver = scipy.sparse.linalg.factorized( scipy.sparse.csc_matrix(LHSMAT))
    #    U = solver(RHS)
    #    U = np.linalg.solve(LHSMAT,RHS)
        U = (scipy.sparse.linalg.spsolve( scipy.sparse.csc_matrix(LHSMAT),RHS, permc_spec="NATURAL") )
    #    U = (scipy.sparse.linalg.bicgstab( scipy.sparse.csc_matrix(LHSMAT),RHS,tol=1e-14) )[0]
        main.uhat[i,0:N2*2/3,k] = U[0::4]#*grid.dealias[0,:,0]
        main.vhat[i,0:N2*2/3,k] = U[1::4]#*grid.dealias[0,:,0]
        main.what[i,0:N2*2/3,k] = U[2::4]#*grid.dealias[0,:,0]
        main.phat[i,0:N2*2/3-1,k] = U[3::4]#*grid.dealias[0,:,0]
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
  I = grid.I#np.eye( (grid.N2/2)*4-1)
  I2 = grid.I2#np.eye( grid.N2/2)
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

  if (main.computeStats == 1 and main.t >= main.tstart_stats):
    main.save_iterations += 1
    main.Ubar[0] += np.mean(main.u,axis=(0,2))
    main.Ubar[1] += np.mean(main.v,axis=(0,2))
    main.Ubar[2] += np.mean(main.w,axis=(0,2))

    main.uubar[0] += np.mean(main.u*main.u ,axis=(0,2))
    main.uubar[1] += np.mean(main.v*main.v,axis=(0,2))
    main.uubar[2] += np.mean(main.w*main.w,axis=(0,2))
    main.uubar[3] += np.mean(main.u*main.v,axis=(0,2))
    main.uubar[4] += np.mean(main.u*main.w,axis=(0,2))
    main.uubar[5] += np.mean(main.v*main.w,axis=(0,2))

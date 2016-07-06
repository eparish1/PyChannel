import threading
import scipy
import sys
import scipy.sparse.linalg
import multiprocessing as mp
import numpy as np
from padding import *

def getA1Mat_Truncate(u):
  N1,N2,N3 = np.shape(u)
  N2 = N2 - 1
  A1 = np.zeros((N2,N2))
  for n in range(0,N2-1):
    for p in range(n+1,N2,2):
      A1[n,p] = 2.*p
  A1[0,:] = A1[0,:] / 2.
  return A1


def getA1Mat(u):
  N1,N2,N3 = np.shape(u)
  A1 = np.zeros((N2,N2))
  for n in range(0,N2-1):
    for p in range(n+1,N2,2):
      A1[n,p] = 2.*p
  A1[0,:] = A1[0,:] / 2.
  return A1

def getA2Mat(u):
  N1,N2,N3 = np.shape(u)
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
  div = np.linalg.norm((uhat_x[:,0:-1,:] + vhat_y[:,0:-1,:] + what_z[:,0:-1,:]))
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

def getRHS_vort(main,grid,myFFT):
  main.uhat = myFFT.dealias(main.uhat)
  main.vhat = myFFT.dealias(main.vhat)
  main.what = myFFT.dealias(main.what)
  main.phat = myFFT.dealias(main.phat)

  u_pad = myFFT.myifft3D_pad(pad(main.uhat,1))
  v_pad = myFFT.myifft3D_pad(pad(main.vhat,1))
  w_pad = myFFT.myifft3D_pad(pad(main.what,1))

  ## compute vorticity
  omegahat_1 = diff_y(main.what) - 1j*grid.k3*main.vhat
  omegahat_2 = 1j*grid.k3*main.uhat - 1j*grid.k1*main.what
  omegahat_3 = 1j*grid.k1*main.vhat - diff_y(main.uhat)

  omega1_pad = myFFT.myifft3D_pad(pad(omegahat_1,1))
  omega2_pad = myFFT.myifft3D_pad(pad(omegahat_2,1))
  omega3_pad = myFFT.myifft3D_pad(pad(omegahat_3,1))

  uu_pad = u_pad*u_pad
  vv_pad = v_pad*v_pad
  ww_pad = w_pad*w_pad

  vom3_pad = v_pad*omega3_pad
  wom2_pad = w_pad*omega2_pad
  uom3_pad = u_pad*omega3_pad
  wom1_pad = w_pad*omega1_pad
  uom2_pad = u_pad*omega2_pad
  vom1_pad = v_pad*omega1_pad


  uuhat = myFFT.dealias( unpad(myFFT.myfft3D_pad(uu_pad),1) )
  vvhat = myFFT.dealias( unpad(myFFT.myfft3D_pad(vv_pad),1) )
  wwhat = myFFT.dealias( unpad(myFFT.myfft3D_pad(ww_pad),1) )

  vom3_hat = myFFT.dealias( unpad(myFFT.myfft3D_pad(vom3_pad),1) )
  wom2_hat = myFFT.dealias( unpad(myFFT.myfft3D_pad(wom2_pad),1)  )
  uom3_hat = myFFT.dealias( unpad(myFFT.myfft3D_pad(uom3_pad),1)  )
  wom1_hat = myFFT.dealias( unpad(myFFT.myfft3D_pad(wom1_pad),1) )
  uom2_hat = myFFT.dealias( unpad(myFFT.myfft3D_pad(uom2_pad),1)  )
  vom1_hat = myFFT.dealias( unpad(myFFT.myfft3D_pad(vom1_pad),1)  )


  vsqrhat = 0.5*( uuhat + vvhat + wwhat)
  uhatRHS_C = -( wom2_hat -vom3_hat + 1j*grid.k1*vsqrhat ) - main.dP ### mean pressure gradient only
  vhatRHS_C = -( uom3_hat -wom1_hat + diff_y(vsqrhat)    ) 
  whatRHS_C = -( vom1_hat -uom2_hat + 1j*grid.k3*vsqrhat )  

 
  main.RHS_explicit[:,0::3,:] = uhatRHS_C[:,:,:]
  main.RHS_explicit[:,1::3,:] = vhatRHS_C[:,:,:]
  main.RHS_explicit[:,2::3,:] = whatRHS_C[:,:,:]
 
  uhat_xx = -grid.k1**2*main.uhat
  uhat_yy = diff_y2(main.uhat)
  uhat_zz = -grid.k3**2*main.uhat

  vhat_xx= -grid.k1**2*main.vhat
  vhat_yy= diff_y2(main.vhat)
  vhat_zz= -grid.k3**2*main.vhat

  what_xx= -grid.k1**2*main.what
  what_yy= diff_y2(main.what)
  what_zz= -grid.k3**2*main.what


  main.RHS_implicit[:,0::3,:] = main.nu*(uhat_xx + uhat_yy + uhat_zz) - 1j*grid.k1*main.phat
  main.RHS_implicit[:,1::3,:] = main.nu*(vhat_xx + vhat_yy + vhat_zz) - diff_y(main.phat)
  main.RHS_implicit[:,2::3,:] = main.nu*(what_xx + what_yy + what_zz) - 1j*grid.k3*main.phat



def lineSolve(main,grid,myFFT,i,I,I2):
  altarray = (-np.ones(grid.N2))**(np.linspace(0,grid.N2-1,grid.N2))
  for k in range(0,grid.N3):
    if (i == 0 and k == 0):
      ### By continuity vhat = 0, don't need to solve for it.
      ### also by definition, p is the fixed pressure gradient, don't need to solve for it. 
      ### Just need to find u and w
      F = np.zeros((grid.N2,grid.N2),dtype='complex')
      F[:,:] =  -main.nu*( grid.A2[:,:]  )
      RHSu = np.zeros((grid.N2),dtype='complex')
      RHSu[:] = main.uhat[i,:,k] + main.dt/2.*(3.*main.RHS_explicit[i,0::3,k] - main.RHS_explicit_old[i,0::3,k]) + \
                main.dt/2.*( main.RHS_implicit[i,0::3,k] )
      RHSw = np.zeros((grid.N2),dtype='complex')
      RHSw[:] = main.what[i,:,k] + main.dt/2.*(3.*main.RHS_explicit[i,2::3,k] - main.RHS_explicit_old[i,2::3,k]) + \
                main.dt/2.*( main.RHS_implicit[i,2::3,k] )

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
      ## SOLUTION VECTOR LOOKS LIKE
      #[ u0,v0,w0,ph0,u1,v1,w1,ph1,...,un,vn,wn]
      ## Form linear matrix for Crank Nicolson terms
      F = np.zeros((grid.N2*4-1,grid.N2*4-1),dtype='complex')
      F[0::4,0::4] = -main.nu*( grid.A2[:,:] - grid.ksqr[i,0,k]*I2[:,:] )###Viscous terms
      F[1::4,1::4] = F[0::4,0::4]  ### put into v eqn as well
      np.fill_diagonal( F[0::4,3::4],1j*grid.k1[i,0,k] )  ## now add pressure to u eqn
      F[1:-2:4,3::4] = grid.A1p[:,:]                      ## v eqn
      np.fill_diagonal( F[2::4,3::4],1j*grid.k3[i,0,k] )  ## w eqn
 
      ## Now create RHS solution vector
      RHS = np.zeros((grid.N2*4-1),dtype='complex')
      RHS[0::4] = main.uhat[i,:,k] +  main.dt/2.*(3.*main.RHS_explicit[i,0::3,k] - main.RHS_explicit_old[i,0::3,k]) + main.dt/2.*main.RHS_implicit[i,0::3,k]
      RHS[1::4] = main.vhat[i,:,k] +  main.dt/2.*(3.*main.RHS_explicit[i,1::3,k] - main.RHS_explicit_old[i,1::3,k]) + main.dt/2.*main.RHS_implicit[i,1::3,k]
      RHS[2::4] = main.what[i,:,k] +  main.dt/2.*(3.*main.RHS_explicit[i,2::3,k] - main.RHS_explicit_old[i,2::3,k]) + main.dt/2.*main.RHS_implicit[i,2::3,k]

      LHSMAT = np.zeros((grid.N2*4-1,grid.N2*4-1),dtype='complex')
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
  
      U = np.linalg.solve(LHSMAT,RHS)
  #    U = (scipy.sparse.linalg.gmres(LHSMAT,RHS, tol=1e-9) )[0]
      main.uhat[i,:,k] = U[0::4]
      main.vhat[i,:,k] = U[1::4]
      main.what[i,:,k] = U[2::4]
      main.phat[i,0:-1,k] = U[3::4]
      main.LHSMAT = LHSMAT
      main.RHS = RHS





def solveBlock(main,grid,myFFT,I,I2,i_start,i_end):
  for i in range(0,grid.N1):
    lineSolve(main,grid,myFFT,i,I,I2)



def advance_AdamsCrank(main,grid,myFFT):
  main.RHS_explicit_old[:,:,:] = main.RHS_explicit[:,:,:]
  getRHS_vort(main,grid,myFFT)
  I = np.eye(grid.N2*4-1)
  I2 = np.eye(grid.N2)
  solveBlock(main,grid,myFFT,I,I2,0,grid.N1)
  #main.uhat = filt2(main.uhat,grid)
  #main.vhat = filt2(main.vhat,grid)

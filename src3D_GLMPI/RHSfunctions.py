import threading
import scipy
import time
import sys
import scipy.sparse.linalg
import multiprocessing as mp
import numpy as np
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

def getRHS_vort(main,grid,myFFT):
  main.uhat = myFFT.dealias(main.uhat)
  main.vhat = myFFT.dealias(main.vhat)
  main.what = myFFT.dealias(main.what)
  main.phat = myFFT.dealias(main.phat)

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


  uuhat = myFFT.dealias( myFFT.myfft3D(uu) )
  vvhat = myFFT.dealias( myFFT.myfft3D(vv) )
  wwhat = myFFT.dealias( myFFT.myfft3D(ww) )

  vom3_hat = myFFT.dealias( myFFT.myfft3D(vom3) )
  wom2_hat = myFFT.dealias( myFFT.myfft3D(wom2) )
  uom3_hat = myFFT.dealias( myFFT.myfft3D(uom3) )
  wom1_hat = myFFT.dealias( myFFT.myfft3D(wom1) )
  uom2_hat = myFFT.dealias( myFFT.myfft3D(uom2) )
  vom1_hat = myFFT.dealias( myFFT.myfft3D(vom1) )


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
    lineSolve(main,grid,myFFT,i,I,I2)



def advance_AdamsCrank(main,grid,myFFT):
  main.RHS_explicit_old[:,:,:] = main.RHS_explicit[:,:,:]
  t1 = time.time() 
  getRHS_vort(main,grid,myFFT)
  I = np.eye(grid.N2*4-1)
  I2 = np.eye(grid.N2)
  t2 = time.time()
  solveBlock(main,grid,myFFT,I,I2,0,grid.N1)


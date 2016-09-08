import threading
import scipy
import time
import sys
import scipy.sparse.linalg
from pylab import *
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
  div = np.zeros(np.shape(main.uhat) ,dtype='complex')
  div[:,0:grid.N2/3*2-1,:] = (uhat_x[:,0:grid.N2/3*2-1,:] + vhat_y[:,0:grid.N2/3*2-1,:] + what_z[:,0:grid.N2/3*2-1,:])
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
  main.uhat = grid.dealias*main.uhat
  main.vhat = grid.dealias*main.vhat
  main.what = grid.dealias*main.what
  main.phat = grid.dealias*main.phat

  myFFT.myifft3D(main.uhat,main.u)
  myFFT.myifft3D(main.vhat,main.u)
  myFFT.myifft3D(main.what,main.u)

  ## compute vorticity
  main.omegahat_1[:] = diff_y(main.what) - 1j*grid.k3*main.vhat
  main.omegahat_2[:] = 1j*grid.k3*main.uhat - 1j*grid.k1*main.what
  main.omegahat_3[:] = 1j*grid.k1*main.vhat - diff_y(main.uhat)

  myFFT.myifft3D(main.omegahat_1,main.omega_1)
  myFFT.myifft3D(main.omegahat_2,main.omega_2)
  myFFT.myifft3D(main.omegahat_3,main.omega_3)

  main.NL[0] = w*omega2
  main.NL[1] = v*omega3
  main.NL[2] = u*omega3
  main.NL[3] = w*omega1
  main.NL[4] = v*omega1
  main.NL[5] = u*omega2
  main.NL[6] = 0.5*(u*u + v*v + w*w)

  for i in range(0,7):
    myFFT.myfft3D(main.NL[i],main.NLhat[i])
    main.NLhat[i] *= grid.dealias
 
  main.RHS_explicit[0] = -( main.NLhat[0] - main.NLhat[1] + 1j*grid.k1*main.NLhat[6] ) - main.dP ### mean pressure gradient only
  main.RHS_explicit[1] = -( main.NLhat[2] - main.NLhat[3] + diff_y(main.NLhat[6])    ) 
  main.RHS_explicit[2] = -( main.NLhat[4] - main.NLhat[5] + 1j*grid.k3*main.NLhat[6] )  

  main.RHS_implicit[0] = main.nu*( -grid.ksqr*main.uhat + diff_y2(main.uhat) ) - 1j*grid.k1*main.phat
  main.RHS_implicit[1] = main.nu*( -grid.ksqr*main.vhat + diff_y2(main.vhat) ) - diff_y(main.phat)
  main.RHS_implicit[2] = main.nu*( -grid.ksrq*main.what + diff_y2(main.what) ) - 1j*grid.k3*main.phat



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
  I = np.eye( (grid.N2*2/3)*4-1)
  I2 = np.eye( grid.N2*2/3)
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

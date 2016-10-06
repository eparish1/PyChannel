import threading
import scipy
import time
import sys
import scipy.sparse.linalg
import multiprocessing as mp
import numpy as np
from padding import separateModes

#import petsc4py, sys
#petsc4py.init(sys.argv)
#from petsc4py import PETSc


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

def checkDivergence2(main,grid):
  ms = np.linspace(0,grid.N2*2/3-1,grid.N2*2/3)
  cm = np.ones(grid.N2*2/3)
  cm[0] = 2.
  div = np.zeros(np.shape(main.uhat) ,dtype='complex')
  div[:,0:grid.N2*2/3-2,:] = 1j*grid.k1[:,0:grid.N2*2/3-2,:]*(cm[None,0:grid.N2*2/3-2,None]*main.uhat[:,0:grid.N2*2/3-2,:] - main.uhat[:,2:grid.N2*2/3,:]) + 2.*ms[None,1:grid.N2*2/3-1,None]*main.vhat[:,1:grid.N2*2/3-1,:] + \
        1j*grid.k3[:,0:grid.N2*2/3-2,:]*(cm[None,0:grid.N2*2/3-2,None]*main.what[:,0:grid.N2*2/3-2,:] - main.what[:,2:grid.N2*2/3,:] ) 
  return div


def checkDivergence(main,grid):
  uhat_x = 1j*grid.k1*main.uhat
  vhat_y = diff_y(main.vhat)
  what_z = 1j*grid.k3*main.what
  div = np.zeros(np.shape(main.uhat) ,dtype='complex')
  div[:,0:grid.N2/3*2-1,:] = (uhat_x[:,0:grid.N2/3*2-1,:] + vhat_y[:,0:grid.N2/3*2-1,:] + what_z[:,0:grid.N2/3*2-1,:])
  #clf()
  #plot(div[1,:,2])
  #ylim([-0.0001,0.0001])
  #pause(0.001)
  #div[:,-4::,:] = 0.
  #div[:,0:2,:] = 0.
  #div[0,:,0] = 0.
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


def getRHS_vort(main,grid,myFFT):
  main.uhat = grid.dealias*main.uhat
  main.vhat = grid.dealias*main.vhat
  main.what = grid.dealias*main.what
  main.phat = grid.dealias*main.phat
  #print(np.linalg.norm(diff_y2(main.uhat) - diff_y2_test(main.uhat) ) )
  myFFT.myifft3D(main.uhat,main.u)
  myFFT.myifft3D(main.vhat,main.v)
  myFFT.myifft3D(main.what,main.w)

  ## compute vorticity
  main.omegahat[0] = diff_y(main.what) - 1j*grid.k3*main.vhat
  main.omegahat[1] = 1j*grid.k3*main.uhat - 1j*grid.k1*main.what
  main.omegahat[2] = 1j*grid.k1*main.vhat - diff_y(main.uhat)

  myFFT.myifft3D(main.omegahat[0],main.omega[0])
  myFFT.myifft3D(main.omegahat[1],main.omega[1])
  myFFT.myifft3D(main.omegahat[2],main.omega[2])

  myFFT.myfft3D(main.w*main.omega[1] - main.v*main.omega[2] , main.NLhat[0] )
  myFFT.myfft3D(main.u*main.omega[2] - main.w*main.omega[0] , main.NLhat[1] )
  myFFT.myfft3D(main.v*main.omega[0] - main.u*main.omega[1] , main.NLhat[2] )
  myFFT.myfft3D(0.5*(main.u*main.u + main.v*main.v + main.w*main.w) , main.NLhat[3]) 
  for i in range(0,4):
    #myFFT.myfft3D(main.NL[i],main.NLhat[i])
    main.NLhat[i] *= grid.dealias
 
  main.RHS_explicit[0] = -( main.NLhat[0] + 1j*grid.k1*main.NLhat[3] ) - main.dP ### mean pressure gradient only
  main.RHS_explicit[1] = -( main.NLhat[1] + diff_y(main.NLhat[3])    ) 
  main.RHS_explicit[2] = -( main.NLhat[2] + 1j*grid.k3*main.NLhat[3] )  

  main.RHS_implicit[0] = main.nu*( -grid.ksqr*main.uhat + diff_y2(main.uhat) ) - 1j*grid.k1*main.phat
  main.RHS_implicit[1] = main.nu*( -grid.ksqr*main.vhat + diff_y2(main.vhat) ) - diff_y(main.phat)
  main.RHS_implicit[2] = main.nu*( -grid.ksqr*main.what + diff_y2(main.what) ) - 1j*grid.k3*main.phat




def lineSolve4(main,grid,myFFT,i,I,I2):
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
        #[ u0 , iu0 , v0 , iv0 , w0 , iw0 , p0 , ip0 ,...]
        #  +0 , +1  , +2 , +3  , +4 , +5  , +6 , +7  ,...]
        ## Form linear matrix for Crank Nicolson terms
        ## Now create RHS solution vector
        RHS =  np.zeros((N2*2/3)*4,dtype='complex')
        RHSu = np.zeros(N2*2/3,dtype='complex')
        RHSv = np.zeros(N2*2/3,dtype='complex')
        RHSw = np.zeros(N2*2/3,dtype='complex')

        RHSu = main.uhat[i,0:N2*2/3,k] +  main.dt/2.*(3.*main.RHS_explicit[0,i,0:N2*2/3,k] - main.RHS_explicit_old[0,i,0:N2*2/3,k]) + main.dt/2.*main.RHS_implicit[0,i,0:N2*2/3,k]
        RHSv = main.vhat[i,0:N2*2/3,k] +  main.dt/2.*(3.*main.RHS_explicit[1,i,0:N2*2/3,k] - main.RHS_explicit_old[1,i,0:N2*2/3,k]) + main.dt/2.*main.RHS_implicit[1,i,0:N2*2/3,k]
        RHSw = main.what[i,0:N2*2/3,k] +  main.dt/2.*(3.*main.RHS_explicit[2,i,0:N2*2/3,k] - main.RHS_explicit_old[2,i,0:N2*2/3,k]) + main.dt/2.*main.RHS_implicit[2,i,0:N2*2/3,k]
  
        ## Now create entire LHS matrix
        A = np.zeros(((grid.N2*2/3)*8,(grid.N2*2/3)*8))

        A[0,0::8] = np.ones(grid.N2*2/3)
        A[1,1::8] = np.ones(grid.N2*2/3)
        A[2,2::8] = np.ones(grid.N2*2/3)
        A[3,3::8] = np.ones(grid.N2*2/3)
        A[4,4::8] = np.ones(grid.N2*2/3)
        A[5,5::8] = np.ones(grid.N2*2/3)
        A[8,0::8] = altarray
        A[9,1::8] = altarray
        A[10,2::8] = altarray
        A[11,3::8] = altarray
        A[12,4::8] = altarray
        A[13,5::8] = altarray

        for j in xrange(2,N2*2/3):
          ## Fill real component of  main diagonal
          val = -main.nu*main.dt*ms[j] -(1.+main.nu*main.dt*grid.ksqr[i,0,k]*0.5)/(2.*ms[j]-2.)  -(1.+main.nu*main.dt*grid.ksqr[i,0,k]*0.5)/(2.*ms[j]+2. ) 
          A[8*j,8*j] = val 
          #A[8j,8j+1] = 0., no imag component
          #A[8j+1,8j] = 0., no imag component
          A[8*j+1,8*j+1] = val
          A[8*j+2,8*j+2] = val #same for v
          A[8*j+3,8*j+3] = val #same for v
          A[8*j+4,8*j+4] = val #same for w
          A[8*j+5,8*j+5] = val #same for v


          ## left real component of left diagonal
          val = cm[(j-2)]/(2.*ms[j] - 2.)*(1. + main.nu*main.dt*grid.ksqr[i,0,k]*0.5) 
          A[8*j,8*(j-2)] = val 
          #A[8j,8(j-2)+1] = 0., no imag component
          #A[8j+1,8(j-2)] = 0., no imag component
          A[8*j+1,8*(j-2)+1] = val
          A[8*j+2,8*(j-2)+2] = val  #same for v
          A[8*j+3,8*(j-2)+3] = val  #same for v
          A[8*j+4,8*(j-2)+4] = val  #same for w
          A[8*j+5,8*(j-2)+5] = val  #same for w

          ## add pressure contribution
          #A[8j,8j+6] = 0. (no real)
          A[8*j,8*j+7] = -main.dt*0.5*grid.k1[i,0,k]*( -1./(2.*ms[j]-2.) - 1./(2.*ms[j]+2.) ) 
          A[8*j,8*(j-2)+7] = -main.dt*0.5*grid.k1[i,0,k]*cm[j-2]/(2.*ms[j]-2.) 
          A[8*j+1,8*j+6] = main.dt*0.5*grid.k1[i,0,k]*( -1./(2.*ms[j]-2.) - 1./(2.*ms[j]+2.) ) 
          A[8*j+1,8*(j-2)+6] = main.dt*0.5*grid.k1[i,0,k]*cm[j-2]/(2.*ms[j]-2.) 

          A[8*j+4,8*j+7] = -main.dt*0.5*grid.k3[i,0,k]*( -1./(2.*ms[j]-2.) - 1./(2.*ms[j]+2.) ) 
          A[8*j+4,8*(j-2)+7] = -main.dt*0.5*grid.k3[i,0,k]*cm[j-2]/(2.*ms[j]-2.) 
          A[8*j+5,8*j+6] = main.dt*0.5*grid.k3[i,0,k]*( -1./(2.*ms[j]-2.) - 1./(2.*ms[j]+2.) ) 
          A[8*j+5,8*(j-2)+6] = main.dt*0.5*grid.k3[i,0,k]*cm[j-2]/(2.*ms[j]-2.) 

        for j in xrange(2,N2*2/3-2):
          ## right diagonal
          val = 1./(2.*ms[j] + 2.)*(1. + main.nu*main.dt*grid.ksqr[i,0,k]*0.5) 
          A[8*j,8*(j+2)] = val
          #A[8j,8(j+2)+1] = 0., no imag component
          #A[8j+1,8(j+2)] = 0., no imag component
          A[8*j+1,8*(j+2)+1] = val 
          A[8*j+2,8*(j+2)+2] = val #same for v
          A[8*j+3,8*(j+2)+3] = val #same for v
          A[8*j+4,8*(j+2)+4] = val #same for w
          A[8*j+5,8*(j+2)+5] = val #same for w

          ## add pressure contribution
          #A[8j,8(j+2)+6] = 0. (no real)
          A[8*j,8*(j+2)+7]   = -main.dt*0.5*grid.k1[i,0,k]*1./(2.*ms[j]+2.) 
          A[8*j+1,8*(j+2)+6] =  main.dt*0.5*grid.k1[i,0,k]*1./(2.*ms[j]+2.) 

          A[8*j+4,8*(j+2)+7] = -main.dt*0.5*grid.k3[i,0,k]*1./(2.*ms[j]+2.) 
          A[8*j+5,8*(j+2)+6] =  main.dt*0.5*grid.k3[i,0,k]*1./(2.*ms[j]+2.) 

        for j in xrange(2,N2*2/3):
          A[8*j+2,8*(j-1)+6] = main.dt*0.5
          A[8*j+3,8*(j-1)+7] = main.dt*0.5
        for j in xrange(2,N2*2/3-1):
          A[8*j+2,8*(j+1)+6] = -main.dt*0.5
          A[8*j+3,8*(j+1)+7] = -main.dt*0.5

        ## continuity equation
        for j in xrange(0,N2*2/3):
          #A[8j+6,8j] = 0. (no real)
          A[8*j+6,8*j+1] = -grid.k1[i,0,k]*cm[j]
          A[8*j+7,8*j] =    grid.k1[i,0,k]*cm[j]
          #A[8j+7,8j+7] = 0. (no real)
          #A[8j+6,8j+4] = 0 (no real)
          A[8*j+6,8*j+5] = -grid.k3[i,0,k]*cm[j]
          A[8*j+7,8*j+4] = grid.k3[i,0,k]*cm[j]
          #A[8j+7,8j+5] = 0 (no real)

        for j in xrange(0,N2*2/3-2):
          #A[8+6,8(j+2)] = 0 (no real)
          A[8*j+6,8*(j+2)+1] =  grid.k1[i,0,k]
          A[8*j+7,8*(j+2)  ] = -grid.k1[i,0,k]
          #A[8+6,8(j+2)+4] = 0 (no real)
          A[8*j+6,8*(j+2)+5] =  grid.k3[i,0,k]
          A[8*j+7,8*(j+2)+4] = -grid.k3[i,0,k]

        for j in xrange(1,N2*2/3):
          A[8*(j-1)+6,8*j+2] = 2.*ms[j]
          A[8*(j-1)+7,8*j+3] = 2.*ms[j]


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

        b = np.zeros((grid.N2*2/3)*8)

        b[::2] = np.real(RHS[:])
        b[1::2]= np.imag(RHS[:])
#
        x = scipy.sparse.linalg.spsolve( scipy.sparse.csc_matrix(A),b, permc_spec="NATURAL")
#
        U = np.zeros((grid.N2*2/3)*4,dtype='complex')
        U[:] = x[::2] + 1j*x[1::2]
        main.uhat[i,0:N2*2/3,k] = U[0::4]#*grid.dealias[0,:,0]
        main.vhat[i,0:N2*2/3,k] = U[1::4]#*grid.dealias[0,:,0]
        main.what[i,0:N2*2/3,k] = U[2::4]#*grid.dealias[0,:,0]
        main.phat[i,0:N2*2/3,k] = U[3::4]#*grid.dealias[0,:,0]



def lineSolve_petsc4py(main,grid,myFFT,i,I,I2):
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
        #[ u0 , iu0 , v0 , iv0 , w0 , iw0 , p0 , ip0 ,...]
        #  +0 , +1  , +2 , +3  , +4 , +5  , +6 , +7  ,...]
        ## Form linear matrix for Crank Nicolson terms
        ## Now create RHS solution vector
        RHS =  np.zeros((N2*2/3)*4,dtype='complex')
        RHSu = np.zeros(N2*2/3,dtype='complex')
        RHSv = np.zeros(N2*2/3,dtype='complex')
        RHSw = np.zeros(N2*2/3,dtype='complex')

        RHSu = main.uhat[i,0:N2*2/3,k] +  main.dt/2.*(3.*main.RHS_explicit[0,i,0:N2*2/3,k] - main.RHS_explicit_old[0,i,0:N2*2/3,k]) + main.dt/2.*main.RHS_implicit[0,i,0:N2*2/3,k]
        RHSv = main.vhat[i,0:N2*2/3,k] +  main.dt/2.*(3.*main.RHS_explicit[1,i,0:N2*2/3,k] - main.RHS_explicit_old[1,i,0:N2*2/3,k]) + main.dt/2.*main.RHS_implicit[1,i,0:N2*2/3,k]
        RHSw = main.what[i,0:N2*2/3,k] +  main.dt/2.*(3.*main.RHS_explicit[2,i,0:N2*2/3,k] - main.RHS_explicit_old[2,i,0:N2*2/3,k]) + main.dt/2.*main.RHS_implicit[2,i,0:N2*2/3,k]
  
        ## Now create entire LHS matrix
        A = PETSc.Mat()
        A.create(PETSc.COMM_WORLD)
        A.setSizes([(grid.N2*2/3)*8,(grid.N2*2/3)*8])
        A.setType('aij') # sparse
        A.setUp()


        A[0,0::8] = np.ones(grid.N2*2/3)
        A[1,1::8] = np.ones(grid.N2*2/3)
        A[2,2::8] = np.ones(grid.N2*2/3)
        A[3,3::8] = np.ones(grid.N2*2/3)
        A[4,4::8] = np.ones(grid.N2*2/3)
        A[5,5::8] = np.ones(grid.N2*2/3)
        A[8,0::8] = altarray
        A[9,1::8] = altarray
        A[10,2::8] = altarray
        A[11,3::8] = altarray
        A[12,4::8] = altarray
        A[13,5::8] = altarray


        for j in xrange(2,N2*2/3):
          ## Fill real component of  main diagonal
          val = -main.nu*main.dt*ms[j] -(1.+main.nu*main.dt*grid.ksqr[i,0,k]*0.5)/(2.*ms[j]-2.)  -(1.+main.nu*main.dt*grid.ksqr[i,0,k]*0.5)/(2.*ms[j]+2. ) 
          A[8*j,8*j] = val 
          #A[8j,8j+1] = 0., no imag component
          #A[8j+1,8j] = 0., no imag component
          A[8*j+1,8*j+1] = val
          A[8*j+2,8*j+2] = val #same for v
          A[8*j+3,8*j+3] = val #same for v
          A[8*j+4,8*j+4] = val #same for w
          A[8*j+5,8*j+5] = val #same for v


          ## left real component of left diagonal
          val = cm[(j-2)]/(2.*ms[j] - 2.)*(1. + main.nu*main.dt*grid.ksqr[i,0,k]*0.5) 
          A[8*j,8*(j-2)] = val 
          #A[8j,8(j-2)+1] = 0., no imag component
          #A[8j+1,8(j-2)] = 0., no imag component
          A[8*j+1,8*(j-2)+1] = val
          A[8*j+2,8*(j-2)+2] = val  #same for v
          A[8*j+3,8*(j-2)+3] = val  #same for v
          A[8*j+4,8*(j-2)+4] = val  #same for w
          A[8*j+5,8*(j-2)+5] = val  #same for w

          ## add pressure contribution
          #A[8j,8j+6] = 0. (no real)
          A[8*j,8*j+7] = -main.dt*0.5*grid.k1[i,0,k]*( -1./(2.*ms[j]-2.) - 1./(2.*ms[j]+2.) ) 
          A[8*j,8*(j-2)+7] = -main.dt*0.5*grid.k1[i,0,k]*cm[j-2]/(2.*ms[j]-2.) 
          A[8*j+1,8*j+6] = main.dt*0.5*grid.k1[i,0,k]*( -1./(2.*ms[j]-2.) - 1./(2.*ms[j]+2.) ) 
          A[8*j+1,8*(j-2)+6] = main.dt*0.5*grid.k1[i,0,k]*cm[j-2]/(2.*ms[j]-2.) 

          A[8*j+4,8*j+7] = -main.dt*0.5*grid.k3[i,0,k]*( -1./(2.*ms[j]-2.) - 1./(2.*ms[j]+2.) ) 
          A[8*j+4,8*(j-2)+7] = -main.dt*0.5*grid.k3[i,0,k]*cm[j-2]/(2.*ms[j]-2.) 
          A[8*j+5,8*j+6] = main.dt*0.5*grid.k3[i,0,k]*( -1./(2.*ms[j]-2.) - 1./(2.*ms[j]+2.) ) 
          A[8*j+5,8*(j-2)+6] = main.dt*0.5*grid.k3[i,0,k]*cm[j-2]/(2.*ms[j]-2.) 

        for j in xrange(2,N2*2/3-2):
          ## right diagonal
          val = 1./(2.*ms[j] + 2.)*(1. + main.nu*main.dt*grid.ksqr[i,0,k]*0.5) 
          A[8*j,8*(j+2)] = val
          #A[8j,8(j+2)+1] = 0., no imag component
          #A[8j+1,8(j+2)] = 0., no imag component
          A[8*j+1,8*(j+2)+1] = val 
          A[8*j+2,8*(j+2)+2] = val #same for v
          A[8*j+3,8*(j+2)+3] = val #same for v
          A[8*j+4,8*(j+2)+4] = val #same for w
          A[8*j+5,8*(j+2)+5] = val #same for w

          ## add pressure contribution
          #A[8j,8(j+2)+6] = 0. (no real)
          A[8*j,8*(j+2)+7]   = -main.dt*0.5*grid.k1[i,0,k]*1./(2.*ms[j]+2.) 
          A[8*j+1,8*(j+2)+6] =  main.dt*0.5*grid.k1[i,0,k]*1./(2.*ms[j]+2.) 

          A[8*j+4,8*(j+2)+7] = -main.dt*0.5*grid.k3[i,0,k]*1./(2.*ms[j]+2.) 
          A[8*j+5,8*(j+2)+6] =  main.dt*0.5*grid.k3[i,0,k]*1./(2.*ms[j]+2.) 

        for j in xrange(2,N2*2/3):
          A[8*j+2,8*(j-1)+6] = main.dt*0.5
          A[8*j+3,8*(j-1)+7] = main.dt*0.5
        for j in xrange(2,N2*2/3-1):
          A[8*j+2,8*(j+1)+6] = -main.dt*0.5
          A[8*j+3,8*(j+1)+7] = -main.dt*0.5

        ## continuity equation
        for j in xrange(0,N2*2/3):
          #A[8j+6,8j] = 0. (no real)
          A[8*j+6,8*j+1] = -grid.k1[i,0,k]*cm[j]
          A[8*j+7,8*j] =    grid.k1[i,0,k]*cm[j]
          #A[8j+7,8j+7] = 0. (no real)
          #A[8j+6,8j+4] = 0 (no real)
          A[8*j+6,8*j+5] = -grid.k3[i,0,k]*cm[j]
          A[8*j+7,8*j+4] = grid.k3[i,0,k]*cm[j]
          #A[8j+7,8j+5] = 0 (no real)

        for j in xrange(0,N2*2/3-2):
          #A[8+6,8(j+2)] = 0 (no real)
          A[8*j+6,8*(j+2)+1] =  grid.k1[i,0,k]
          A[8*j+7,8*(j+2)  ] = -grid.k1[i,0,k]
          #A[8+6,8(j+2)+4] = 0 (no real)
          A[8*j+6,8*(j+2)+5] =  grid.k3[i,0,k]
          A[8*j+7,8*(j+2)+4] = -grid.k3[i,0,k]

        for j in xrange(1,N2*2/3):
          A[8*(j-1)+6,8*j+2] = 2.*ms[j]
          A[8*(j-1)+7,8*j+3] = 2.*ms[j]

        A.assemblyBegin()
        A.assemblyEnd()


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


        x, b = A.getVecs()

        b[::2] = np.real(RHS[:])
        b[1::2]= np.imag(RHS[:])

        x.set(0)
        ## create linear solver
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        # use conjugate gradients
        ksp.setType('cg')
        # and incomplete Cholesky
        ksp.getPC().setType('gasm')
        ksp.setOperators(A)
        ksp.setFromOptions()
        ksp.solve(b, x)
        U = np.zeros((grid.N2*2/3)*4,dtype='complex')
        U[:] = x[::2] + 1j*x[1::2]
        main.uhat[i,0:N2*2/3,k] = U[0::4]#*grid.dealias[0,:,0]
        main.vhat[i,0:N2*2/3,k] = U[1::4]#*grid.dealias[0,:,0]
        main.what[i,0:N2*2/3,k] = U[2::4]#*grid.dealias[0,:,0]
        main.phat[i,0:N2*2/3,k] = U[3::4]#*grid.dealias[0,:,0]



def lineSolve3(main,grid,myFFT,i,I,I2):
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
  
        A = np.zeros(((grid.N2*2/3)*4,(grid.N2*2/3)*4),dtype='complex')
        A[0,0::4] = 1.
        A[1,1::4] = 1.
        A[2,2::4] = 1.
        A[4,0::4] = altarray
        A[5,1::4] = altarray
        A[6,2::4] = altarray

        #np.fill_diagonal(LHSMATu[2::,2::],-main.nu*main.dt*ms[2::] \
       #                                -(1.+main.nu*main.dt*grid.ksqr[i,0,k]*0.5)/(2.*ms[2::]-2.) \
      #                                 -(1.+main.nu*main.dt*grid.ksqr[i,0,k]*0.5)/(2.*ms[2::]+2.) )
       # np.fill_diagonal(LHSMATu[2::,0:-2],cm[0:-2]/(2.*ms[2::]-2.)*(1. + main.nu*main.dt*grid.ksqr[i,0,k]*0.5) )
       # np.fill_diagonal(LHSMATu[2::,4::],       1./(2.*ms[2::]+2.)*(1. + main.nu*main.dt*grid.ksqr[i,0,k]*0.5) )

        for j in xrange(2,N2*2/3):
          ## Fill main diagonal
          A[4*j,4*j] =  -main.nu*main.dt*ms[j] -(1.+main.nu*main.dt*grid.ksqr[i,0,k]*0.5)/(2.*ms[j]-2.)  -(1.+main.nu*main.dt*grid.ksqr[i,0,k]*0.5)/(2.*ms[j]+2. ) 
          A[4*j+1,4*j+1] = A[4*j,4*j] #same for v
          A[4*j+2,4*j+2] = A[4*j,4*j] #same for w


          ## left diagonal
          A[4*j,4*(j-2)] = cm[(j-2)]/(2.*ms[j] - 2.)*(1. + main.nu*main.dt*grid.ksqr[i,0,k]*0.5) 
          A[4*j+1,4*(j-2)+1] = A[4*j,4*(j-2)] #same for v
          A[4*j+2,4*(j-2)+2] = A[4*j,4*(j-2)] #same for w

          ## add pressure contribution
          A[4*j,4*j+3] = main.dt*0.5*1j*grid.k1[i,0,k]*( -1./(2.*ms[j]-2.) - 1./(2.*ms[j]+2.) ) 
          A[4*j,4*(j-2)+3] = main.dt*0.5*1j*grid.k1[i,0,k]*cm[j-2]/(2.*ms[j]-2.) 
          A[4*j+2,4*j+3] = main.dt*0.5*1j*grid.k3[i,0,k]*( -1./(2.*ms[j]-2.) - 1./(2.*ms[j]+2.) ) 
          A[4*j+2,4*(j-2)+3] = main.dt*0.5*1j*grid.k3[i,0,k]*cm[j-2]/(2.*ms[j]-2.) 


        for j in xrange(2,N2*2/3-2):
          ## right diagonal
          A[4*j,4*(j+2)] = 1./(2.*ms[j] + 2.)*(1. + main.nu*main.dt*grid.ksqr[i,0,k]*0.5) 

          A[4*j+1,4*(j+2)+1] = A[4*j,4*(j+2)] #same for v
          A[4*j+2,4*(j+2)+2] = A[4*j,4*(j+2)] #same for w

          ## add pressure contribution
          A[4*j,4*(j+2)+3] = main.dt*0.5*1j*grid.k1[i,0,k]*1./(2.*ms[j]+2.) 
          A[4*j+2,4*(j+2)+3] = main.dt*0.5*1j*grid.k3[i,0,k]*1./(2.*ms[j]+2.) 

        for j in xrange(2,N2*2/3):
          A[4*j+1,4*(j-1)+3] = main.dt*0.5
        for j in xrange(2,N2*2/3-1):
          A[4*j+1,4*(j+1)+3] = -main.dt*0.5

        ## continuity equation
        for j in xrange(0,N2*2/3):
          A[4*j+3,4*j] = 1j*grid.k1[i,0,k]*cm[j]
          A[4*j+3,4*j+2] = 1j*grid.k3[i,0,k]*cm[j]
        for j in xrange(0,N2*2/3-2):
          A[4*j+3,4*(j+2)] = -1j*grid.k1[i,0,k]
          A[4*j+3,4*(j+2)+2] = -1j*grid.k3[i,0,k]
        for j in xrange(1,N2*2/3):
          A[4*(j-1)+3,4*j+1] = 2.*ms[j]
        ##========== Continuity Equation ================
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

        #Na1,Na2 = np.shape(A)
        #A2 = np.zeros((Na1*2,Na2*2))
        #A2[0::2,0::2] = np.real(A)
        #A2[0::2,1::2] = -np.imag(A)
        #A2[1::2,0::2] = np.imag(A)
        #A2[1::2,1::2] = np.real(A)
        #RHS2 = np.zeros((2*Na1))
        #RHS2[0::2] = np.real(RHS)
        #RHS2[1::2] = np.imag(RHS)
        #U2 = np.zeros(2*Na1)
        #U2 = scipy.sparse.linalg.spsolve( scipy.sparse.csc_matrix(A2),RHS2, permc_spec="NATURAL")
        #U = U2[0::2] + 1j*U2[1::2]

        ## Now create entire LHS matrix
        A2 = PETSc.Mat()
        A2.create(PETSc.COMM_WORLD)
        A2.setSizes([N,N])
        A2.setType('aij') # sparse
        A2.setUp()
        #Na1,Na2 = np.shape(A)
        #A2 = np.zeros((Na1*2,Na2*2))
        #A2[0::2,0::2] = np.real(A)
        #A2[0::2,1::2] = -np.imag(A)
        #A2[1::2,0::2] = np.imag(A)
        #A2[1::2,1::2] = np.real(A)


        A2.assemblyBegin()
        A2.assemblyEnd()

        xr, br = A2.getVecs()
        xi, bi = A2.getVecs()

        br[:] = real(RHS[:])
        bi[:] = imag(RHS[:])
        xr.set(0)
        xi.set(0)
        ## create linear solver
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        # use conjugate gradients
        ksp.setType('cg')
        # and incomplete Cholesky
        #ksp.getPC().setType('icc')
        ksp.setOperators(A2)
        ksp.setFromOptions()
        ksp.solve(br, xr)
        ksp.solve(br, xi)


    #    solver = scipy.sparse.linalg.factorized( scipy.sparse.csc_matrix(LHSMAT))
    #    U = solver(RHS)
    #    U = np.linalg.solve(LHSMAT,RHS)
    #    U = (scipy.sparse.linalg.spsolve( scipy.sparse.csc_matrix(A),RHS, permc_spec="NATURAL") )
    #     U = (scipy.sparse.linalg.gmres( scipy.sparse.csc_matrix(A),RHS,tol=1e-8) )[0]
        main.uhat[i,0:N2*2/3,k] = U[0::4]#*grid.dealias[0,:,0]
        main.vhat[i,0:N2*2/3,k] = U[1::4]#*grid.dealias[0,:,0]
        main.what[i,0:N2*2/3,k] = U[2::4]#*grid.dealias[0,:,0]
        main.phat[i,0:N2*2/3,k] = U[3::4]#*grid.dealias[0,:,0]
        #print(1j*grid.k1[i,0,k]*main.uhat[i,N2*2/3-2,k] + main.vhat[i,N2*2/3-1,k]*2.*(N2*2/3-1) + 1j*grid.k3[i,0,k]*main.what[i,N2*2/3-2,k] ) 
       # print(np.linalg.norm(dot(LHSMAT,U)[3::4] ) )
       # print(LHSMAT[-4,0::-7])
        main.LHSMAT = A
       # main.RHS = RHS



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
    #    U = (scipy.sparse.linalg.gmres( scipy.sparse.csc_matrix(LHSMAT),RHS,tol=1e-8) )[0]
        main.uhat[i,0:N2*2/3,k] = U[0::4]#*grid.dealias[0,:,0]
        main.vhat[i,0:N2*2/3,k] = U[1::4]#*grid.dealias[0,:,0]
        main.what[i,0:N2*2/3,k] = U[2::4]#*grid.dealias[0,:,0]
        main.phat[i,0:N2*2/3,k] = U[3::4]#*grid.dealias[0,:,0]
        #print(1j*grid.k1[i,0,k]*main.uhat[i,N2*2/3-2,k] + main.vhat[i,N2*2/3-1,k]*2.*(N2*2/3-1) + 1j*grid.k3[i,0,k]*main.what[i,N2*2/3-2,k] ) 
       # print(np.linalg.norm(dot(LHSMAT,U)[3::4] ) )
       # print(LHSMAT[-4,0::-7])
        main.LHSMAT = LHSMAT
       # main.RHS = RHS



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
       lineSolve2(main,grid,myFFT,i,I,I2)



def advance_AdamsCrank(main,grid,myFFT):
  main.RHS_explicit_old[:,:,:] = main.RHS_explicit[:,:,:]
  t1 = time.time() 
  main.getRHS(main,grid,myFFT)
  I = np.eye( (grid.N2*2/3)*4-1)
  I2 = np.eye( grid.N2*2/3)
  t2 = time.time()
  solveBlock(main,grid,myFFT,I,I2,0,grid.N1)
  if (main.computeStats == 1):
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




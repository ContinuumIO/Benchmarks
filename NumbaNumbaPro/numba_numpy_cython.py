import numpy as np
from numbapro.vectorize import basic, parallel, stream
from numbapro.vectorize import Vectorize
from numba.decorators import jit
from numba import *
from numpy import sqrt
from numpy import log
from numpy import exp
f, d = f4, f8


mu = .1

from diffusion_cython import cy_update

def runCythonDiffusion():

    u_in = np.zeros([Lx,Ly],dtype=np.double)
    u_in[Lx/2,Ly/2] = 1000.0
    tempU = np.zeros([Lx, Ly], dtype=np.double)
    for i in range(iterNum):
        cy_update(u_in,tempU)


@jit(argtypes=[double[:,:], double[:,:], int32])
def diffuseNumba(u,tempU,iterNum):
    row = u.shape[0]
    col = u.shape[1]
    
    for n in range(iterNum):
        for i in range(1,row-1):
            for j in range(1,col-1):
                tempU[i,j] = u[i,j] + mu * (u[i+1,j]-2*u[i,j]+u[i-1,j] + \
                                            u[i,j+1]-2*u[i,j]+u[i,j-1] )
        for i in range(1,row-1):
            for j in range(1,col-1):
                u[i,j] = tempU[i,j]
                tempU[i,j] = 0.0

def diffusePurePython(u,tempU,iterNum):
    row = u.shape[0]
    col = u.shape[1]
    
    for n in range(iterNum):
        for i in range(1,row-1):
            for j in range(1,col-1):
                tempU[i,j] = u[i,j] + mu * (u[i+1,j]-2*u[i,j]+u[i-1,j] + \
                                            u[i,j+1]-2*u[i,j]+u[i,j-1] )
        for i in range(1,row-1):
            for j in range(1,col-1):
                u[i,j] = tempU[i,j]
                tempU[i,j] = 0.0

def diffuseNumPy(cen, cenXM, cenXP, cenYM, cenYP,do_diffusion):
    return cen+do_diffusion*mu*(cenXP-2.0*cen+cenXM+cenYP-2.0*cen+cenYM)
   
def runNumbaDiffusion():

    u_in = np.zeros([Lx,Ly],dtype=np.double)
    u_in[Lx/2,Ly/2] = 1000.0
    tempU = np.zeros([Lx, Ly], dtype=np.double)
    for i in range(iterNum):
        diffuseNumba(u_in,tempU,1)

def runPurePythonDiffusion():

    u_in = np.zeros([Lx,Ly],dtype=np.double)
    u_in[Lx/2,Ly/2] = 1000.0
    tempU = np.zeros([Lx, Ly], dtype=np.double)
    for i in range(iterNum):
        diffusePurePython(u_in,tempU,1)

def runNumPyDiffusion():
    u_in = np.zeros([Lx,Ly],dtype=np.double)
    u_in[Lx/2,Ly/2] = 1000.0
    do_diffusion = np.ones([Lx,Ly],dtype=np.float64)[1:-1,1:-1]

    
    for i in range(iterNum):
        cen = u_in[1:-1,1:-1] #center (everything but the top-bottom left-right edge)

        cenYP = u_in[2:,1:-1] #center+1y (shift center square down)
        cenYM = u_in[:-2,1:-1] #center-1y (shift center square up)

        cenXP = u_in[1:-1, 2:] #center+1x (shift center square right)
        cenXM = u_in[1:-1, :-2] #center+1x (shift center square right)
        u_in[1:-1, 1:-1] = diffuseNumPy(cen, cenXM, cenXP, cenYM, cenYP,do_diffusion)


if __name__ == '__main__':
    global SIZE
    global Lx, Ly
    global iternum
    import timeit
    
    lim = 13
    iterNum = 1000
    
    numpyTimes = np.zeros((lim,3))
    purePythonTimes = np.zeros((lim,3))
    cythonTimes = np.zeros((lim,3))
    numbaTimes = np.zeros((lim,3))
    
    for power in range(lim):
        Lx,Ly = (2 ** power, 2 ** power)
  
        print '\tN\t', Lx, power        

        t = timeit.Timer("runNumbaDiffusion()", "from __main__ import runNumbaDiffusion")

        print 'Run NumbaDiffusion :'
        time =  t.repeat(5, 1)
        numbaTimes[power] = (Lx,np.mean(time), np.std(time))
        
        t = timeit.Timer("runNumPyDiffusion()", "from __main__ import runNumPyDiffusion")

        print 'Run NumPyDiffusion:'
        time =  t.repeat(5, 1)
        numpyTimes[power] = (Lx,np.mean(time), np.std(time))

        t = timeit.Timer("runCythonDiffusion()", "from __main__ import runCythonDiffusion")

        print 'Run CythonDiffusion:'
        time =  t.repeat(5, 1)
        cythonTimes[power] = (Lx,np.mean(time), np.std(time))

        # t = timeit.Timer("runPurePythonDiffusion()", "from __main__ import runPurePythonDiffusion")

        # print '\runPurePythonDiffusion with runPurePythonDiffusion:'
        # time =  t.repeat(5, 1)
        # purePythonTimes[power] = (Lx,np.mean(time), np.std(time))

from pylab import *


figure(figsize=(9,6))
# plot(numpyTimes[:,0],numpyTimes[:,1],'s--',label="NumPy")
plot(cythonTimes[:,0],cythonTimes[:,1],'^--',label="Cython")
plot(numbaTimes[:,0],numbaTimes[:,1],'o--',label="Numba")
legend(loc=2)
grid(True)
ylabel('Time in Seconds')

xlabel('Size of N for NxN Matrix')
title(str(iterNum)+' iterations of: Laplace\'s Equation with Size N  NxN Matrix')
savefig('numba_cython-laplace-9x6.png')
# savefig('numpy_numba_cython-laplace-9x6.png')




# numbaTimes = np.array([[  1.00000000e+00,   3.77769470e-03,   2.92913983e-04],
#        [  2.00000000e+00,   3.61261368e-03,   1.33212344e-04],
#        [  4.00000000e+00,   3.69253159e-03,   9.86298904e-05],
#        [  8.00000000e+00,   4.05659676e-03,   1.03543364e-04],
#        [  1.60000000e+01,   6.39796257e-03,   4.54201122e-04],
#        [  3.20000000e+01,   1.99831486e-02,   4.31255638e-04],
#        [  6.40000000e+01,   7.33105659e-02,   1.27010678e-03],
#        [  1.28000000e+02,   2.79729033e-01,   1.29021972e-03],
#        [  2.56000000e+02,   1.13603315e+00,   2.54988797e-02],
#        [  5.12000000e+02,   4.74292784e+00,   2.67772190e-01],
#        [  1.02400000e+03,   2.78597854e+01,   9.87763841e-01],
#        [  2.04800000e+03,   8.13986894e+01,   4.81671079e-01],
#        [  4.09600000e+03,   3.01944600e+02,   1.90879245e+00]])

# cythonTimes = np.array([[  1.00000000e+00,   1.14998817e-03,   1.53380066e-05],
#        [  2.00000000e+00,   1.18560791e-03,   2.12711065e-05],
#        [  4.00000000e+00,   1.23648643e-03,   2.88474264e-05],
#        [  8.00000000e+00,   1.64899826e-03,   1.33451379e-05],
#        [  1.60000000e+01,   3.73063087e-03,   8.43932532e-05],
#        [  3.20000000e+01,   1.57917976e-02,   2.74106455e-04],
#        [  6.40000000e+01,   6.28121853e-02,   2.49787840e-04],
#        [  1.28000000e+02,   2.55983782e-01,   1.26665345e-04],
#        [  2.56000000e+02,   1.02570543e+00,   4.54792288e-04],
#        [  5.12000000e+02,   4.62424855e+00,   3.30890594e-02],
#        [  1.02400000e+03,   2.76604324e+01,   1.18625669e-02],
#        [  2.04800000e+03,   7.40540400e+01,   1.76062163e-01],
#        [  4.09600000e+03,   2.97999561e+02,   6.30987773e-02]])

# numpyTimes = np.array([[  1.00000000e+00,   3.60116005e-02,   2.28695327e-04],
#        [  2.00000000e+00,   3.80225182e-02,   1.33358063e-04],
#        [  4.00000000e+00,   6.97188377e-02,   2.11547181e-04],
#        [  8.00000000e+00,   7.22037792e-02,   5.63207113e-04],
#        [  1.60000000e+01,   8.22724342e-02,   5.73320663e-04],
#        [  3.20000000e+01,   9.85260487e-02,   2.79729861e-04],
#        [  6.40000000e+01,   1.70972776e-01,   2.31113206e-04],
#        [  1.28000000e+02,   4.69114256e-01,   8.13398566e-04],
#        [  2.56000000e+02,   3.16633196e+00,   5.24392911e-02],
#        [  5.12000000e+02,   1.88155525e+01,   5.36689859e-02],
#        [  1.02400000e+03,   8.61007920e+01,   3.23209449e-01],
#        [  2.04800000e+03,   3.52033693e+02,   1.54928561e-01],
#        [  4.09600000e+03,   1.45333962e+03,   6.30987773e-02]])
cimport numpy as np

def cy_update(np.ndarray[double, ndim=2] u,np.ndarray[double, ndim=2] tempU):
    cdef unsigned int i, j
    for i in xrange(1,u.shape[0]-1):
        for j in xrange(1, u.shape[1]-1):
            tempU[i,j] = u[i,j] + 0.1 * (u[i+1,j]-2*u[i,j]+u[i-1,j] + \
                                        u[i,j+1]-2*u[i,j]+u[i,j-1] )

    for i in xrange(1,u.shape[0]-1):
        for j in xrange(1, u.shape[1]-1):
            u[i,j] = tempU[i,j]
            tempU[i,j] = 0.0
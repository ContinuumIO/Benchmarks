import numpy as np
from numbapro.vectorize import basic, parallel, stream
from numbapro.vectorize import Vectorize
from numba.decorators import jit
from numba import *
from numpy import sqrt
from numpy import log
from numpy import exp
f, d = f4, f8



import pyfits

def logThreshold(img_mat):
    return 10*log(img_mat+10)+10


bv = Vectorize(logThreshold, backend='bytecode')
bv.add(restype=f, argtypes=[f])
b_poly_d = bv.build_ufunc()


pv = Vectorize(logThreshold, target='parallel', backend='bytecode')
pv.add(restype=f, argtypes=[f])
p_poly_d = pv.build_ufunc()

    



f1 = pyfits.getdata('fpC-003836-r3-0254.fit')
iterNum = 10

def threshNumbaPro():
    [b_poly_d(f1) for i in range(iterNum)]

def threshNumbaProParallel():
    [p_poly_d(f1) for i in range(iterNum)]

def threshNumPy():
    [logThreshold(f1) for i in range(iterNum)]



if __name__ == '__main__':


    import timeit

    t = timeit.Timer("threshNumbaPro()", "from __main__ import threshNumbaPro")

    print 'threshNumbaPro with NumbaPro BasicVectorize:'
    time =  t.repeat(5, 1)
    numbaProTime = (iterNum,np.mean(time), np.std(time))

    
    t = timeit.Timer("threshNumbaProParallel()", "from __main__ import threshNumbaProParallel")

    print 'threshNumbaProParallel with NumbaPro Parallel:'
    time =  t.repeat(5, 1)
    numbaProParallelTime = (iterNum,np.mean(time), np.std(time))

    t = timeit.Timer("threshNumPy()", "from __main__ import threshNumPy")

    print 'threshNumPy with NumPy:'

    time =  t.repeat(5, 1)
    numpyTime = (iterNum,np.mean(time), np.std(time))



from pylab import *


figure(figsize=(9,6))

space_width = 0.15
col_width = 0.1

labels = ["NumPy", "BasicVectorize", "ParallelVectorize"]
times = np.array([numpyTime,numbaProTime,numbaProParallelTime])

xlocations = np.linspace(0,1,3)+col_width
bar(xlocations , times[:,1], col_width, yerr=times[:,2], color='blue')

xticks(xlocations+col_width/2,labels,size='medium',rotation=15)

xlim(0,xlocations[-1]+col_width*2)
# yticks(tick_range, size='medium')
# ylabel('Memory Size in MBs')
ylabel('Time in Seconds')
title("Time to Log Threshold Fit Image from SDSS")
# suptitle('Faster Speeds are Smaller',style='italic', fontsize=12)

# formatter = FixedFormatter([str(x) for x in tick_range])
# gca().yaxis.set_major_formatter(formatter)
gca().yaxis.grid(which='major')
gca().get_xaxis().tick_bottom()
# savefig(output_name)
# show()
savefig('log_thresh_numbapro.png')


# from pylab import *
# errorbar(numpyTimes[:,0],numpyTimes[:,1],yerr=numpyTimes[:,2],label="NumPy")
# # errorbar(purePythonTimes[:,0],purePythonTimes[:,1],yerr=purePythonTimes[:,2],label="Pure Python")
# errorbar(cythonTimes[:,0],cythonTimes[:,1],yerr=cythonTimes[:,2],label="Cython")
# errorbar(numbaTimes[:,0],numbaTimes[:,1],yerr=numbaTimes[:,2],label="Numba")
# legend(loc=2)
# grid(True)
# ylabel('Time in Seconds')
# # xlabel('Size of N for 1D Matrix')
# xlabel('Size of N for NxN Matrix')
# title(str(iterNum)+' iterations of: Laplace\'s Equation with Size N  NxN Matrix')

# http://data.sdss3.org/sas/dr9/boss/photo/data/3629/fields/
# http://data.sdss3.org/sas/dr9/boss/photo/data/273/fields/4/
# http://data.sdss3.org/sas/dr9/boss/photo/data/211/fields/4/
# http://data.sdss3.org/sas/dr9/
# http://homepages.inf.ed.ac.uk/rbf/HIPR2/pixlog.htm
# http://packages.python.org/pyfits/users_guide/users_image.html
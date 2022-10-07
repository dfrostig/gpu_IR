from astropy.io import fits 
import numpy as np
import cupy as cp
from cupyx.profiler import benchmark
import argparse
import time

# load in data and format for reduction  
def load_data_w(file_path):
    
    # pull in data
    hdul = fits.open(file_path)
    data = hdul[0].data
    hdr = hdul[0].header
    
    if(hdr['NAXIS']!=3):
        raise Exception("Need to check formatting. TODO: Improve") 
    else:
        x = hdr['NAXIS1']
        y = hdr['NAXIS2']
        z = hdr['NAXIS3']
        
    # reshape into a line, x*y long   
    data = data.reshape((z, x*y)).T
    # add one dimension for matrix operations later
    data = data[:,:,np.newaxis]
    # retrun data and shape
    print(data.shape)
    return data.T, (x,y,z) 

# load in data and format for reduction  
def load_data(file_path):
    
    # pull in data
    hdul = fits.open(file_path)
    data = hdul[1].data
    hdr = hdul[1].header

    if(hdr['NAXIS']!=4):
        raise Exception("Need to check formatting. TODO: Improve") 
    else:
        x = hdr['NAXIS1']
        y = hdr['NAXIS2']
        z = hdr['NAXIS3']
            
        
    # reshape into a line, x*y long   
    data = data.reshape((1, z, x*y))
    # retrun data and shape
    return data, (x,y,z) 

# Ordinary least squares fit for cpu
def cpu_ols(A,y):
    return np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)   

# Ordinary least squares fit for gpu
def gpu_ols_3D(A,y):
    return cp.dot((cp.dot(cp.linalg.inv(cp.dot(A.T,A)),A.T)),y)


    
# Run through OLS fit for cpu
def linearfit_cpu(data, fnum, n_loops, n_pix, slope_array):
    # prepare A matrix for OLS fit
    A = np.vstack([fnum, np.ones(len(fnum))]).T
    
    # loop over data batches
    for i in range(n_loops):
        res = cpu_ols(A, data[:,:,i*n_pix:(i*n_pix)+n_pix])
        slope_array[i*n_pix:(i*n_pix)+n_pix] = res[0].flatten()  
    return 
        
# Run through OLS fit for cpu
def linearfit_gpu(data, fnum, n_loops, n_pix, slope_array):
    # prepare A matrix for OLS fit
    A = cp.vstack([fnum, np.ones(len(fnum))]).T
    
    # loop over data batches
    for i in range(n_loops):
        res = cpu_ols(A, cp.asarray(data[:,:,i*n_pix:(i*n_pix)+n_pix]))
        slope_array[i*n_pix:(i*n_pix)+n_pix] = res[0].get().flatten()  
    return 

# Run through OLS fit for cpu
def polyfit_gpu(data, fnum, n_loops, n_pix, slope_array):
    # loop over data batches
    for i in range(n_loops):
        image_gpu = cp.asarray(data[:,i*n_pix:(i*n_pix)+n_pix])
        slope_array[i*n_pix:(i*n_pix)+n_pix] = cp.polyfit(fnum,image_gpu, 1)[0].get()
    return slope_array

# Benchmark  for cpu
def benchmark_cpu(data, data_shape):
    cpu_times= []
    
    # get data shape
    pix_tot = data_shape[0]*data_shape[1]
    n_frames = data_shape[2]
    
    # pass in data in batches of 2**n, up to the total pixel number
    data_batches = range(2,int(np.ceil(np.log2(pix_tot))),1)
    
    # prepare frame range and result array outside of loop
    fnum = np.linspace(0,n_frames-1, n_frames)
    slope_array = np.zeros(int(pix_tot))
    
    # loop through data batch sizes
    for batch_size in data_batches:
        n_pix = 2**(batch_size)
        n_loops = int(np.ceil(pix_tot/(n_pix)))

        try:
            t1 = time.time()
            linearfit_cpu(data, fnum, n_loops, n_pix, slope_array)
            t2 = time.time()
            cpu_times.append((n_pix, t2-t1))
        except:
            print("Hit maximum data size, aborting...")
    min_time = min(cpu_times, key = lambda t: t[1])
    print(cpu_times)
    print("Minimum time was", min_time[1], " seconds for batches of", 
          min_time[0]," pixels")
    return cpu_times
    
# Benchmark  for gpu
def benchmark_gpu(data, data_shape):
    gpu_benchmarks = []
    
    # get data shape
    pix_tot = data_shape[0]*data_shape[1]
    n_frames = data_shape[2]
    
    # pass in data in batches of 2**n, up to the total pixel number
    data_batches = range(8,int(np.ceil(np.log2(pix_tot))),1)
    slope_array = np.zeros(int(pix_tot))
    
    # prepare frame range and result array outside of loop
    fnum = np.linspace(0,n_frames-1, n_frames)
    slope_array = np.zeros(int(pix_tot))
    
    
    # loop through data batch sizes
    for batch_size in data_batches:
        n_pix = 2**(batch_size)
        n_loops = int(np.ceil(pix_tot/(n_pix)))

        try:
            print("Benchmarking ", n_pix, n_loops)
            b = benchmark(linearfit_gpu, 
                          (data, fnum, n_loops, n_pix, slope_array,), 
                          n_repeat=5, n_warmup=2)
            gpu_benchmarks.append(b)
        except:
            print("Hit maximum data size, aborting...")
            
    
    min_cpu = []
    min_gpu = []
    for i in range(len(gpu_benchmarks)):
        min_cpu.append(np.min(gpu_benchmarks[i].cpu_times))
        min_gpu.append(np.min(gpu_benchmarks[i].gpu_times))
     
    print("Minimum times: ", min_gpu)
    min_time_idx = min_gpu.index(min(min_gpu))
    print("Minimum time was", min(min_gpu), " for batches of", 
          2**data_batches[min_time_idx]," pixels")
    return gpu_benchmarks

if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description="Reduce IR data with a linear fit",
                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # input file
    parser.add_argument("-i", "--input", dest="filename", required=True,
                    help="input file", metavar="FILE")
    # method: cpu or gpu
    parser.add_argument("-c", "--cpu", dest="cpu", action='store_true',
                help="select cpu implementation")
    parser.add_argument("-g", "--gpu", dest="gpu", action='store_true',
                help="select gpu implementation")

    
    args = parser.parse_args()
    
    if args.cpu is None and args.gpu is None:
        parser.error("selecting either --gpu or --cpu is required")
    
    in_path = args.filename
    
    # load data
    data, data_shape = load_data(in_path)
    
    # run reduction
    if args.cpu:
        benchmark_cpu(data, data_shape)
    elif args.gpu:
        benchmark_gpu(data, data_shape)

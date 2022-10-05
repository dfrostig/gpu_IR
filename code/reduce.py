from astropy.io import fits 
import numpy as np
import cupy as cp
import argparse

# load in data and format for reduction  
def load_data(file_path):
    
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
    return data, (x,y,z) 

# Ordinary least squares fit for cpu
def cpu_ols(A,y):
    return np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)   

# Ordinary least squares fit for gpu
def gpu_ols_3D(A,y):
    return cp.dot((cp.dot(cp.linalg.inv(cp.dot(A.T,A)),A.T)),y)
    
# Run through OLS fit for cpu
def linearfit_cpu(data, n_frames, n_loops, n_pix, slope_array):
    # range of frame numbers
    fnum = np.linspace(0,n_frames-1, n_frames)
    
    # prepare A matrix for OLS fit
    A = np.vstack([fnum, np.ones(len(fnum))]).T
    
    # loop over data batches
    for i in range(n_loops):
        res = cpu_ols(A, data[i*n_pix:(i*n_pix)+n_pix,:,:])
        slope_array[i*n_pix:(i*n_pix)+n_pix] = res[0].flatten()  
    return slope_array
        
# Run through OLS fit for cpu
def linearfit_gpu(data, n_frames, n_loops, n_pix, slope_array):
    # range of frame numbers
    fnum = np.linspace(0,n_frames-1, n_frames)
    
    # prepare A matrix for OLS fit
    A = cp.vstack([fnum, np.ones(len(fnum))]).T
    
    # loop over data batches
    for i in range(n_loops):
        res = cpu_ols(A, cp.asarray(data[i*n_pix:(i*n_pix)+n_pix,:,:]))
        slope_array[i*n_pix:(i*n_pix)+n_pix] = res[0].get().flatten()  
    return slope_array

# Save data 
def save_results(data, data_shape, n_pix, file_path, method):
    # get data shape
    (pix_tot, n_frames, m_axis) = data.shape
    
    # pepare array for results
    slope_array = np.zeros(int(pix_tot))
    
    # set number of data batches based on pixel size
    n_loops = int(np.ceil(pix_tot/(n_pix)))
    
    # do the fit
    if method == "cpu":
        linearfit_cpu(data, n_frames, n_loops, n_pix, slope_array)
    else:
        linearfit_gpu(data, n_frames, n_loops, n_pix, slope_array)
    
    # save
    hdu = fits.PrimaryHDU()
    hdu.data = slope_array.reshape(data_shape[1],data_shape[0])
    hdu.writeto(file_path, overwrite=True)
    print("Saved reduced data to ", file_path)
    

if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description="Reduce IR data with a linear fit",
                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # input files
    parser.add_argument("-i", "--input", dest="filename", required=True,
                    help="input file", metavar="FILE")
    # output file
    parser.add_argument("-o", "--output", dest="outfile", required=True,
                help="output filename and path", metavar="FILE")
    # method: cpu or gpu
    parser.add_argument("-c", "--cpu", dest="cpu", action='store_true',
                help="select cpu implementation")
    parser.add_argument("-g", "--gpu", dest="gpu", action='store_true',
                help="select gpu implementation")
    # get optimal batch size in pixels
    parser.add_argument("-n", "--number_of_pixels", dest="npix", default=2**16,
                type=int, help="number of pixels in batch", metavar="NPIX")
    
    args = parser.parse_args()
    
    if args.cpu is None and args.gpu is None:
        parser.error("selecting either --gpu or --cpu is required")
    
    in_path = args.filename
    out_path = args.outfile
    
    # load data
    data, data_shape = load_data(in_path)
    
    # run reduction
    if args.cpu:
        save_results(data, data_shape, 2**8, out_path, "cpu")
    elif args.gpu:
        save_results(data, data_shape, 2**8, out_path, "gpu")
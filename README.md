# gpu_IR
A public repo for GPU implementations of IR astronomical data reduction techniques.

## How to use gpu_IR

gpu_IR is currently base on two scripts, `benchmark.py` and `reduce.py`. Both scripts take in a fits file of non-destructive or up-the-ramp data and performs a linear fit on each pixel. `benchmark.py` reduces (linear fit) the data on either a gpu or cpu, splitting up the job into smaller batches of pixels. The script runs through a range of batch sizes and reports the fastest data reduction time and the corresponding batch size. Then one can run `reduce.py` to reduce data with the optimal batch size as an input.


## Running `benchmark.py`

Example
` python benchmark.py -i "data.fits" --cpu `
 
 Help:

 usage: benchmark.py [-h] -i FILE [-c] [-g]

Reduce IR data with a linear fit

optional arguments:

  -h, --help            show this help message and exit
  
  -i FILE, --input FILE
  
                        input file (default: None)
                        
  -c, --cpu             select cpu implementation (default: False)
  
  -g, --gpu             select gpu implementation (default: False)
  
  
  
## Running `reduce.py`

` python reduce.py -i "data.fits" -o "slopes.fits" --gpu -n 256 `


usage: reduce.py [-h] -i FILE -o FILE [-c] [-g] [-n NPIX]

Reduce IR data with a linear fit

optional arguments:
  -h, --help            show this help message and exit
  -i FILE, --input FILE
                        input file (default: None)
  -o FILE, --output FILE
                        output filename and path (default: None)
  -c, --cpu             select cpu implementation (default: False)
  -g, --gpu             select gpu implementation (default: False)
  -n NPIX, --number_of_pixels NPIX
                        number of pixels in batch (default: 65536)

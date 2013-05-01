To compile the extension:

.../ornlm> python setup.py build_ext --inplace

To run the example :

.../ornlm> python test_filters.py

It reports the maximum difference with respect to the matlab's output for each filter, 
itshould be less than 10^-11:

Maximum error [ornlm (block size= 3x3)]:  2.27373675443e-13
Maximum error [ornlm (block size= 5x5)]:  2.27373675443e-13
Maximum error [hsm]:  1.02318153949e-12
Maximum error [ascm]:  6.8212102633e-13

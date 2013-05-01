cdef extern from "ornlm.h":
    void ornlm(double *image, int nslices, int nrows, int ncols, int v, int patchLen, double h, double *filtered)
    void boxFilter(double *image, unsigned char *mask, int nslices, int nrows, int ncols, double *filtered)
cdef extern from "upfirdn.h":
    void firdn_matrix(double *F, int n, int m, double *h, int len, double *out)
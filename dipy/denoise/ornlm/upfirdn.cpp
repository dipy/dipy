/*! \file upfirdn.cpp
	\author Omar Ocegueda
	\brief Implementation of required wavelet functions to implement Pierrick's filter
*/
#include "upfirdn.h"
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)<(b))?(b):(a))
/*! \brief special case of matlab's "upfirdn": Applies filter h to f and downsamples by 2
*/
void firdn_vector(double *f, int n, int stride_f, double *h, int len, double *out, int stride_out){
	int outLen=(n+len)/2;
	for(int i=0, x=0, ox=0;i<outLen;++i, x+=2, ox+=stride_out){
		double sum=0;
		int limInf=MAX(0, x-len+1);
		int limSup=MIN(n-1, x);
		for(int k=limInf, ks=limInf*stride_f;k<=limSup;++k, ks+=stride_f){
			sum+=f[ks]*h[x-k];
		}
		out[ox]=sum;
	}
}

/*! \brief special case of matlab's "upfirdn": upsamples f by 2 and applies filter h to the upsampled f 
*/
void upfir_vector(double *f, int n, int stride_f, double *h, int len, double *out, int stride_out){
	int outLen=2*n+len-2;
	for(int x=0, ox=0;x<outLen;++x, ox+=stride_out){
		int limInf=MAX(0, x-len+1);
		if(limInf&1){
			++limInf;
		}
		int limSup=MIN(2*(n-1), x);
		if(limSup&1){
			--limSup;
		}
		double sum=0;
		for(int k=limInf, ks=(limInf*stride_f)/2;k<=limSup;k+=2, ks+=stride_f){
			sum+=f[ks]*h[x-k];
		}
		out[ox]=sum;
	}
}

/*! \brief special case of matlab's "upfirdn": applies firdn to each column of F
*/
void firdn_matrix(double *F, int n, int m, double *h, int len, double *out){
	for(int j=0;j<m;++j){
		firdn_vector(F+j, n, m, h, len, out+j, m);
	}
}

/*! \brief special case of matlab's "upfirdn": applies upfir to each column of F
*/
void upfir_matrix(double *F, int n, int m, double *h, int len, double *out){
	for(int j=0;j<m;++j){
		upfir_vector(F+j, n, m, h, len, out+j, m);
	}
}



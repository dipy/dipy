/*! \file wavelet.h
	\author Omar Ocegueda
	\brief Implementation of required wavelet functions to implement Pierrick's filter
*/
#ifndef WAVELET_H
#define WAVELET_H
void firdn_matrix(double *F, int n, int m, double *h, int len, double *out);
void upfir_matrix(double *F, int n, int m, double *h, int len, double *out);
#endif

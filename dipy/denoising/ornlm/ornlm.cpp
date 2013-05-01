/* Pierrick Coupe - pierrick.coupe@gmail.com                               */
/* Jose V. Manjon - jmanjon@fis.upv.es                                     */
/* Addapted to Python by Omar Ocegueda - jomaroceguedag@gmail.com          */
/* Brain Imaging Center, Montreal Neurological Institute.                  */
/* Mc Gill University                                                      */
/*                                                                         */
/* Copyright (C) 2008 Pierrick Coupe and Jose V. Manjon                    */


/***************************************************************************
*              3D Adaptive Multiresolution Non-Local Means Filter          *
* Pierrick Coupe a, Jose V. Manjon, Montserrat Robles , D. Louis Collins   *
***************************************************************************/


/*                          Details on ONLM filter                        */
/***************************************************************************
 *  The ONLM filter is described in:                                       *
 *                                                                         *
 *  P. Coupé, P. Yger, S. Prima, P. Hellier, C. Kervrann, C. Barillot.     *
 *  An Optimized Blockwise Non Local Means Denoising Filter for 3D Magnetic*
 *  Resonance Images. IEEE Transactions on Medical Imaging, 27(4):425-441, * 
 *  Avril 2008                                                             *
 ***************************************************************************/


/*                      Details on Rician adaptation                      */
/***************************************************************************
 *  The adaptation to Rician noise is described in:                        *
 *                                                                         *
 *  N. Wiest-Daesslé, S. Prima, P. Coupé, S.P. Morrissey, C. Barillot.     *
 *  Rician noise removal by non-local means filtering for low              *
 *  signal-to-noise ratio MRI: Applications to DT-MRI. In 11th             *
 *  International Conference on Medical Image Computing and                *
 *  Computer-Assisted Intervention, MICCAI'2008,                           *
 *  Pages 171-179, New York, États-Unis, Septembre 2008                    *
 ***************************************************************************/

#include "math.h"
#include <stdlib.h>
/* Function which compute the weighted average for one block */
void Average_block(double *ima,int x,int y,int z,int neighborhoodsize,double *average, double weight, int* vol_size)
{
int x_pos,y_pos,z_pos;
bool is_outside;

int a,b,c;

int count = 0;

	for (c = 0; c<(2*neighborhoodsize+1);c++)
	{
		for (b = 0; b<(2*neighborhoodsize+1);b++)
		{
			for (a = 0; a<(2*neighborhoodsize+1);a++)
			{
	
				is_outside = false;
				x_pos = x+a-neighborhoodsize;
				y_pos = y+b-neighborhoodsize;
				z_pos = z+c-neighborhoodsize;
	
				if ((z_pos < 0) || (z_pos > vol_size[2]-1)) is_outside = true;
				if ((y_pos < 0) || (y_pos > vol_size[0]-1)) is_outside = true;
				if ((x_pos < 0) || (x_pos > vol_size[1]-1)) is_outside = true;
		
				if (is_outside)
					average[count] = average[count] + ima[z*(vol_size[0]*vol_size[1])+(x*vol_size[0])+y]*ima[z*(vol_size[0]*vol_size[1])+(x*vol_size[0])+y]*weight;
				else	
					average[count] = average[count] + ima[z_pos*(vol_size[0]*vol_size[1])+(x_pos*vol_size[0])+y_pos]*ima[z_pos*(vol_size[0]*vol_size[1])+(x_pos*vol_size[0])+y_pos]*weight;
				
				count++;
			}
		}
	}
}

/* Function which computes the value assigned to each voxel */
void Value_block(double *Estimate, double *Label,int x,int y,int z,int neighborhoodsize,double *average, double global_sum, int* vol_size, double hh)
{
int x_pos,y_pos,z_pos;
int ret;
bool is_outside;
double value = 0.0;
double denoised_value =0.0;
double label = 0.0;
int count=0 ;
int a,b,c;

	for (c = 0; c<(2*neighborhoodsize+1);c++)
	{
		for (b = 0; b<(2*neighborhoodsize+1);b++)
		{
			for (a = 0; a<(2*neighborhoodsize+1);a++)
			{
	
	
				is_outside = false;
				x_pos = x+a-neighborhoodsize;
				y_pos = y+b-neighborhoodsize;
				z_pos = z+c-neighborhoodsize;
	
				if ((z_pos < 0) || (z_pos > vol_size[2]-1)) is_outside = true;
				if ((y_pos < 0) || (y_pos > vol_size[0]-1)) is_outside = true;
				if ((x_pos < 0) || (x_pos > vol_size[1]-1)) is_outside = true;
				if (!is_outside)
				{
		

					value = Estimate[z_pos*(vol_size[0]*vol_size[1])+(x_pos*vol_size[0])+y_pos];
					denoised_value  = (average[count]/global_sum) - hh;
					if (denoised_value > 0)
						denoised_value = sqrt(denoised_value);
					else denoised_value = 0.0;
					
					value = value + denoised_value;
					
					label = Label[(y_pos + x_pos*vol_size[0] + z_pos *vol_size[0] * vol_size[1])];
					Estimate[z_pos*(vol_size[0]*vol_size[1])+(x_pos*vol_size[0])+y_pos] = value;
					Label[(y_pos + x_pos*vol_size[0] + z_pos *vol_size[0] * vol_size[1])] = label +1;
		
				}
				count++;
			}
		}
	}
}


double distance(double* ima,int x,int y,int z,int nx,int ny,int nz,int f,int sx,int sy,int sz)
{

double d,acu,distancetotal,inc;
int i,j,k,ni1,nj1,ni2,nj2,nk1,nk2,kk;

acu=0;
distancetotal=0;
	
for(k=-f;k<=f;k++)
{
	for(i=-f;i<=f;i++)
	{
		for(j=-f;j<=f;j++)
		{
			ni1=x+i;
			nj1=y+j;
			nk1=z+k;
			ni2=nx+i;
			nj2=ny+j;
			nk2=nz+k;
			
			if(ni1<0) ni1=-ni1;
			if(nj1<0) nj1=-nj1;
			if(ni2<0) ni2=-ni2;
			if(nj2<0) nj2=-nj2;
			if(nk1<0) nk1=-nk1;
			if(nk2<0) nk2=-nk2;
			
			if(ni1>=sx) ni1=2*sx-ni1-1;
			if(nj1>=sy) nj1=2*sy-nj1-1;
			if(nk1>=sz) nk1=2*sz-nk1-1;
			if(ni2>=sx) ni2=2*sx-ni2-1;
			if(nj2>=sy) nj2=2*sy-nj2-1;
			if(nk2>=sz) nk2=2*sz-nk2-1;
			
			distancetotal = distancetotal + ((ima[nk1*(sx*sy)+(ni1*sy)+nj1]-ima[nk2*(sx*sy)+(ni2*sy)+nj2])*(ima[nk1*(sx*sy)+(ni1*sy)+nj1]-ima[nk2*(sx*sy)+(ni2*sy)+nj2]));
			acu=acu + 1;
		}
	}
}

d=distancetotal/acu;

return d;

}

void ornlm(double *ima, int ndim, int *dims, int v, int f, double h, double *fima){
double label,estimate,epsilon,mu1,var1,w,totalweight,wmax,d,mean,var,t1,t2,hh;
int init,Ndims,i,j,k,ii,jj,kk,ni,nj,nk,indice,nvox;
hh=2*h*h;
Ndims = pow((2*f+1),ndim);
nvox=dims[0]*dims[1]*dims[2];

/*Allocate memory and assign output pointer*/
double *average=(double*) malloc(Ndims*sizeof(double));
double *means=(double*) malloc(nvox*sizeof(double));
double *varianzas=(double*) malloc(nvox*sizeof(double));
double *Estimate =(double*) malloc(nvox*sizeof(double));
double *Label=(double*) malloc(nvox*sizeof(double));


for (i = 0; i < dims[2] *dims[1] * dims[0];i++)
{
	Estimate[i] = 0.0;
	Label[i] = 0.0;
	fima[i] = 0.0;
}


for(k=0;k<dims[2];k++)
{
	for(i=0;i<dims[1];i++)
	{
		for(j=0;j<dims[0];j++)
		{
			mean=0;
			indice=0;
			for(ii=-1;ii<=1;ii++)
			{
				for(jj=-1;jj<=1;jj++)
				{
					for(kk=-1;kk<=1;kk++)
					{
						ni=i+ii;
							nj=j+jj;		   		  
						nk=k+kk;
						
						if(ni<0) ni=-ni;
						if(nj<0) nj=-nj;
						if(nk<0) nk=-nk;
						if(ni>=dims[1]) ni=2*dims[1]-ni-1;
						if(nj>=dims[0]) nj=2*dims[0]-nj-1;
						if(nk>=dims[2]) nk=2*dims[2]-nk-1;
						
								
						mean = mean + ima[nk*(dims[0]*dims[1])+(ni*dims[0])+nj];
						indice=indice+1;                
					
					}
				}
			}
			mean=mean/indice;
			means[k*(dims[0]*dims[1])+(i*dims[0])+j]=mean;
		}
	}
}

for(k=0;k<dims[2];k++)
{
	for(i=0;i<dims[1];i++)
	{
		for(j=0;j<dims[0];j++)
		{
			var=0;
			indice=0;
			for(ii=-1;ii<=1;ii++)
			{
				for(jj=-1;jj<=1;jj++)
				{
					for(kk=-1;kk<=1;kk++)
					{
						ni=i+ii;
						nj=j+jj;		   		  
						nk=k+kk;		   		  
							if(ni>=0 && nj>=0 && nk>0 && ni<dims[1] && nj<dims[0] && nk<dims[2])
							{
							var = var + (ima[nk*(dims[0]*dims[1])+(ni*dims[0])+nj]-means[k*(dims[0]*dims[1])+(i*dims[0])+j])*(ima[nk*(dims[0]*dims[1])+(ni*dims[0])+nj]-means[k*(dims[0]*dims[1])+(i*dims[0])+j]);
							indice=indice+1;
							}
					}
				}
			}
			var=var/(indice-1);
			varianzas[k*(dims[0]*dims[1])+(i*dims[0])+j]=var;
		}
	}
}

/*filter*/

epsilon = 0.00001;
mu1 = 0.95;
var1 = 0.5;
init = 0;

for(k=0;k<dims[2];k+=2)
{
	for(i=0;i<dims[1];i+=2)
	{
		for(j=0;j<dims[0];j+=2)
		{
		
			for (init=0 ; init < Ndims; init++)
			{
				average[init]=0.0;
			}
			/*average=0;*/

			totalweight=0.0;
			
		
		
			if ((means[k*(dims[0]*dims[1])+(i*dims[0])+j])>epsilon && (varianzas[k*(dims[0]*dims[1])+(i*dims[0])+j]>epsilon))
			{
				wmax=0.0;
				
				for(kk=-v;kk<=v;kk++)
				{
					for(ii=-v;ii<=v;ii++)
					{
						for(jj=-v;jj<=v;jj++)
						{
							ni=i+ii;
							nj=j+jj;
							nk=k+kk;

							if(ii==0 && jj==0 && kk==0) continue; 
				
							if(ni>=0 && nj>=0 && nk>=0 && ni<dims[1] && nj<dims[0] && nk<dims[2])
							{
					
				
								if ((means[nk*(dims[0]*dims[1])+(ni*dims[0])+nj])> epsilon && (varianzas[nk*(dims[0]*dims[1])+(ni*dims[0])+nj]>epsilon))
								{
				
									t1 = (means[k*(dims[0]*dims[1])+(i*dims[0])+j])/(means[nk*(dims[0]*dims[1])+(ni*dims[0])+nj]);  
									t2 = (varianzas[k*(dims[0]*dims[1])+(i*dims[0])+j])/(varianzas[nk*(dims[0]*dims[1])+(ni*dims[0])+nj]);
	
									if(t1>mu1 && t1<(1/mu1) && t2>var1 && t2<(1/var1))
									{                 
										
										d=distance(ima,i,j,k,ni,nj,nk,f,dims[1],dims[0],dims[2]);
	
										w = exp(-d/(h*h));
	
										if(w>wmax) wmax = w;
										
										Average_block(ima,ni,nj,nk,f,average,w,dims);
										
										/*average = average + w*ima[nk*(dims[0]*dims[1])+(ni*dims[0])+nj];*/
										totalweight = totalweight + w;
									}
								}
			
			
			
							}
						}
					}

					
		
				}
				
				if(wmax==0.0) wmax=1.0;
		
				/*average = average + wmax*ima[k*(dims[0]*dims[1])+(i*dims[0])+j];*/
				Average_block(ima,i,j,k,f,average,wmax,dims);
					
				totalweight = totalweight + wmax;
	
				/*fima[k*(dims[0]*dims[1])+(i*dims[0])+j] = average / totalweight;*/ 
					 
				if(totalweight != 0.0)
				Value_block(Estimate,Label,i,j,k,f,average,totalweight,dims,hh);
		
		
			}
		
				else
				{
					wmax=1.0;
					Average_block(ima,i,j,k,f,average,wmax,dims);
					totalweight = totalweight + wmax;
					Value_block(Estimate,Label,i,j,k,f,average,totalweight,dims,hh);
				}
			
		}
	} 
}

label = 0.0;
estimate = 0.0;


/* Aggregation of the estimators (i.e. means computation) */
for (k = 0; k < dims[2]; k++ )
{
	for (i = 0; i < dims[1]; i++ )
	{
		for (j = 0; j < dims[0]; j++ )
		{
			label = Label[k*(dims[0]*dims[1])+(i*dims[0])+j];
			if (label == 0.0)
			{
				fima[k*(dims[0]*dims[1])+(i*dims[1])+j] = ima[k*(dims[0]*dims[1])+(i*dims[0])+j];
	
			}
			else
			{
				estimate = Estimate[k*(dims[0]*dims[1])+(i*dims[0])+j];
				estimate = (estimate/label);
				fima[k*(dims[0]*dims[1])+(i*dims[0])+j]=estimate;
	
			}
		}
	}
}

free(average);
free(means);
free(varianzas);
free(Estimate);
free(Label);
return;

}


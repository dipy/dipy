/* dipy math functions
 *
 * To give some platform independence for simple math functions
 */

#include <math.h>
#include "numpy/npy_math.h"

#define dpy_isnan npy_isnan
#define dpy_log2 npy_log2
#define DPY_PI NPY_PI

/* From numpy npy_math.c.src */
#ifndef HAVE_LOG2
double dpy_log2(double x)
{
    return NPY_LOG2E*log(x);
}
#endif

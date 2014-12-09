/* dipy math functions
 *
 * To give some platform independence for simple math functions
 */

#include <math.h>
#include "numpy/npy_math.h"

#define dpy_isnan npy_isnan
#define dpy_signbit npy_signbit
#define DPY_PI NPY_PI

/* From numpy npy_math.c.src */
#ifndef HAVE_LOG2
double dpy_log2(double x)
{
    return NPY_LOG2E*log(x);
}
#else
#define dpy_log2 npy_log2
#endif

/* From numpy npy_math.c.src */
#ifndef HAVE_RINT
double dpy_rint(double x)
{
    double y, r;

    y = npy_floor(x);
    r = x - y;

    if (r > 0.5) {
        y += 1.0;
    }

    /* Round to nearest even */
    if (r == 0.5) {
        r = y - 2.0*npy_floor(0.5*y);
        if (r == 1.0) {
            y += 1.0;
        }
    }
    return y;
}
#else
#define dpy_rint npy_rint
#endif

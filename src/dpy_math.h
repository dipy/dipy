/* dipy math functions
 *
 * To give some platform independence for simple math functions
 */

#include <math.h>
#include "numpy/npy_math.h"

#define DPY_PI NPY_PI

/* From numpy npy_math.c.src commit b2f6792d284b0e9383093c30d51ec3a82e8312fd*/
double dpy_log2(double x)
{
#ifdef HAVE_LOG2
    return log2(x);
#else
    return NPY_LOG2E*log(x);
#endif
}

#if (defined(_WIN32) || defined(_WIN64)) && !defined(__GNUC__)
#define fmin min
#endif

#define dpy_floor(x) floor((double)(x))

double dpy_rint(double x)
{
#ifdef HAVE_RINT
    return rint(x);
#else
    double y, r;

    y = dpy_floor(x);
    r = x - y;

    if (r > 0.5) {
        y += 1.0;
    }

    /* Round to nearest even */
    if (r == 0.5) {
        r = y - 2.0*dpy_floor(0.5*y);
        if (r == 1.0) {
            y += 1.0;
        }
    }
    return y;
#endif
}


int dpy_signbit(double x)
{
#ifdef signbit
    return signbit(x);
#else
    union
    {
        double d;
        short s[4];
        int i[2];
    } u;

    u.d = x;

#if NPY_SIZEOF_INT == 4

#ifdef WORDS_BIGENDIAN /* defined in pyconfig.h */
    return u.i[0] < 0;
#else
    return u.i[1] < 0;
#endif

#else  /* NPY_SIZEOF_INT != 4 */

#ifdef WORDS_BIGENDIAN
    return u.s[0] < 0;
#else
    return u.s[3] < 0;
#endif

#endif  /* NPY_SIZEOF_INT */

#endif /*NPY_HAVE_DECL_SIGNBIT*/
}


#ifndef NPY_HAVE_DECL_ISNAN
    #define dpy_isnan(x) ((x) != (x))
#else
    #ifdef _MSC_VER
        #define dpy_isnan(x) _isnan((x))
    #else
        #define dpy_isnan(x) isnan(x)
    #endif
#endif


#ifndef NPY_HAVE_DECL_ISFINITE
    #ifdef _MSC_VER
        #define dpy_isfinite(x) _finite((x))
    #else
        #define dpy_isfinite(x) !npy_isnan((x) + (-x))
    #endif
#else
    #define dpy_isfinite(x) isfinite((x))
#endif


#ifndef NPY_HAVE_DECL_ISINF
    #define dpy_isinf(x) (!dpy_isfinite(x) && !dpy_isnan(x))
#else
    #ifdef _MSC_VER
        #define dpy_isinf(x) (!_finite((x)) && !_isnan((x)))
    #else
        #define dpy_isinf(x) isinf((x))
    #endif
#endif

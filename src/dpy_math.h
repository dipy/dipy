/* dipy math functions
 *
 * To give some platform independence for simple math functions
 */

#include <math.h>
#include "numpy/npy_math.h"

#define DPY_PI NPY_PI

/* If HAVE_LOG2 is not defined, then the replacement
'npy_log2' was defined, and we can safely link to it,
otherwise, we can simply call log2*/
#if 0
#ifdef HAVE_LOG2
#define npy_log2 log2
#endif

#ifdef HAVE_RINT
#define npy_rint rint
#endif
#endif

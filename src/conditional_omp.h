/* Header file to conditionally wrap omp.h defines
 *
 * _OPENMP should be defined if omp.h is safe to include
 */
#if defined(_OPENMP)
#include <omp.h>
#define have_openmp 1
#else
/* These are fake defines to make these symbols valid in the c / pyx file
 *
 * All uses of these symbols should to be prefaced with ``if have_openmp``, as
 * in:
 *
 *     cdef omp_lock_t lock
 *     if have_openmp:
 *         openmp.omp_init_lock(&lock)
 *
 * */
typedef int omp_lock_t;
void omp_init_lock(omp_lock_t *lock) {};
void omp_destroy_lock(omp_lock_t *lock) {};
void omp_set_lock(omp_lock_t *lock) {};
void omp_unset_lock(omp_lock_t *lock) {};
int omp_test_lock(omp_lock_t *lock) {};
void omp_set_dynamic(int dynamic_threads) {};
void omp_set_num_threads(int num_threads) {};
int omp_get_num_procs() {};
int omp_get_max_threads() {};
#define have_openmp 0
#endif

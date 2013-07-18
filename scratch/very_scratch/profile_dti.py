# Profiling by fitting an actual, rather sizeable data-set.

import time
import numpy as np
import dipy.data as dpd
import dipy.reconst.dti as dti
reload(dti)

img, gtab = dpd.read_stanford_hardi()

t1 = time.time()
dm_ols = dti.TensorModel(gtab, fit_method='OLS')
fit_ols = dm_ols.fit(img.get_data())
t2 = time.time()
print("Done with OLS. That took %s seconds to run"%(t2-t1))

dm_nlls = dti.TensorModel(gtab, fit_method='NLLS')
fit_nlls = dm_nlls.fit(img.get_data())
t3 = time.time()
print("Done with NLLS. That took %s seconds to run"%(t3-t2))

dm_restore = dti.TensorModel(gtab, fit_method='restore', sigma=10)
fit_restore = dm_restore.fit(img.get_data())
t4 = time.time()
print("Done with RESTORE. That took %s seconds to run"%(t4-t3))

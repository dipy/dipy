import math
from numba import njit


def trilinear_interp_generator(DIMX, DIMY, DIMZ, DIMT):
    @njit
    def trilinear_interp(pmf_volume, point, out_pmf):
        px = point[0]
        py = point[1]
        pz = point[2]
    
        # Bounds check  (matches: point.x < -0.5 || point.x+0.5 >= DIMX ...)
        if px < -0.5 or px + 0.5 >= DIMX:
            return -1
        if py < -0.5 or py + 0.5 >= DIMY:
            return -1
        if pz < -0.5 or pz + 0.5 >= DIMZ:
            return -1
    
        # Floor coordinates
        flx = math.floor(px)
        fly = math.floor(py)
        flz = math.floor(pz)
    
        # Weights along each axis
        wx1 = px - flx          # high-side weight
        wx0 = 1.0 - wx1         # low-side weight
    
        wy1 = py - fly
        wy0 = 1.0 - wy1
    
        wz1 = pz - flz
        wz0 = 1.0 - wz1
    
        # Clamped voxel indices (matches MAX(0,fl) / MIN(DIM-1, ...))
        ix0 = max(0, int(flx))
        ix1 = min(DIMX - 1, ix0 + 1)
    
        iy0 = max(0, int(fly))
        iy1 = min(DIMY - 1, iy0 + 1)
    
        iz0 = max(0, int(flz))
        iz1 = min(DIMZ - 1, iz0 + 1)
    
        # Accumulate the 8-corner trilinear blend for every ODF direction
        # Matches the triple loop:  for i in {0,1}: for j in {0,1}: for k in {0,1}:
        #   out[t] += wgh[0][i]*wgh[1][j]*wgh[2][k] * dataf[ix,iy,iz,t]
        for t in range(DIMT):
            out_pmf[t] = (
                wx0 * wy0 * wz0 * pmf_volume[ix0, iy0, iz0, t] +
                wx0 * wy0 * wz1 * pmf_volume[ix0, iy0, iz1, t] +
                wx0 * wy1 * wz0 * pmf_volume[ix0, iy1, iz0, t] +
                wx0 * wy1 * wz1 * pmf_volume[ix0, iy1, iz1, t] +
                wx1 * wy0 * wz0 * pmf_volume[ix1, iy0, iz0, t] +
                wx1 * wy0 * wz1 * pmf_volume[ix1, iy0, iz1, t] +
                wx1 * wy1 * wz0 * pmf_volume[ix1, iy1, iz0, t] +
                wx1 * wy1 * wz1 * pmf_volume[ix1, iy1, iz1, t]
            )
    
        return 0
    return trilinear_interp

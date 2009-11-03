import numpy as np
import dipy.core.track_metrics as tm

xyz = np.array([[0,0,0],[20,20,0],[30,10,0],[40,20,0],[50,0,0]])

print xyz

for i in range(3):
    for j in range(i+2,5):
        print (i,j), 'nopar', tm.MDL_nopar(xyz[i:j])
        print (i,j), ' par',  tm.MDL_par(xyz[i:j])
        for k in range(i+1,j):
            print (i,j,k-1,k), 'perp', tm.lee_perpendicular_distance(xyz[i],xyz[j],xyz[k-1],xyz[k])
            print (i,j,k-1,k), 'angl', tm.lee_angle_distance(xyz[i],xyz[j],xyz[k-1],xyz[k])
    
xyz = np.array([[0,0,0],[20,5,0],[30,12,0],[40,5,0],[50,0,0]])

print xyz

for i in range(3):
    for j in range(i+2,5):
        print (i,j), 'nopar', tm.MDL_nopar(xyz[i:j])
        print (i,j), ' par',  tm.MDL_par(xyz[i:j])
        for k in range(i+1,j):
            print (i,j,k-1,k), 'perp', tm.lee_perpendicular_distance(xyz[i],xyz[j],xyz[k-1],xyz[k])
            print (i,j,k-1,k), 'angl', tm.lee_angle_distance(xyz[i],xyz[j],xyz[k-1],xyz[k])
    

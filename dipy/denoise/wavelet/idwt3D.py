from sfb3D import sfb3D
def idwt3D(w, J, sf):
    y=w[J];
    for k in range(J)[::-1]:
        y=sfb3D(y, w[k], sf, sf, sf);
    return y

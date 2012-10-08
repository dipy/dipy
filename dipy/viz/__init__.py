# Init file for visualization package


# We make the visualization requirements (mayavi, matplotlib) optional
# imports:

try: 
    # Mayavi appears in many guises:
    try:
        from enthought.mayavi import mlab
    except ImportError:
        from mayavi import mlab
    has_mayavi = True
except ImportError:
    e_s = "You do not have Mayavi installed. Some visualization functions"
    e_s += " might not work."
    print(e_s)
    has_mayavi = False

try:
    import matplotlib
    has_mpl = True
except ImportError:
    e_s = "You do not have Matplotlib installed. Some visualization functions"
    e_s += " might not work for you."
    print(e_s)
    has_mpl = False

if has_mayavi:
    from ._show_odfs import show_odfs

if has_mpl:
    import projections

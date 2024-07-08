import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "utils",
    ],
    submod_attrs={
        "dpy": ["Dpy"],
        "gradients": ["read_bvals_bvecs"],
        "pickles": ["load_pickle", "save_pickle"],
    },
)

__all__ = ["read_bvals_bvecs", "Dpy", "save_pickle", "load_pickle", "utils"]

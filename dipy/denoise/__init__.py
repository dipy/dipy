# init for denoise aka the denoising module

import lazy_loader as lazy

__getattr__, __lazy_dir__, _ = lazy.attach_stub(__name__, __file__)

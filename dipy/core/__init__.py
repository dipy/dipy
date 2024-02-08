# Init for core dipy objects
""" Core objects """

import lazy_loader as lazy

__getattr__, __lazy_dir__, _ = lazy.attach_stub(__name__, __file__)

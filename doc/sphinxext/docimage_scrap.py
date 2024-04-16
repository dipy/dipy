import glob
import os
import shutil
import time

from sphinx_gallery.scrapers import figure_rst


class ImageFileScraper:
    def __init__(self):
        """Scrape image files that are already present in current folder."""
        self.embedded_images = {}
        self.start_time = time.time()

    def __call__(self, block, block_vars, gallery_conf):
        # Find all image files in the current directory.
        path_example = os.path.dirname(block_vars['src_file'])
        image_files = _find_images(path_example)
        # Iterate through files, copy them to the SG output directory
        image_names = []
        image_path_iterator = block_vars['image_path_iterator']
        for path_orig in image_files:
            # If we already know about this image and it hasn't been modified
            # since starting, then skip it
            mod_time = os.stat(path_orig).st_mtime
            already_embedded = (path_orig in self.embedded_images and
                                mod_time <= self.embedded_images[path_orig])
            existed_before_build = mod_time <= self.start_time
            if already_embedded or existed_before_build:
                continue
            # Else, we assume the image has just been modified and is displayed
            path_new = next(image_path_iterator)
            self.embedded_images[path_orig] = mod_time
            image_names.append(path_new)
            shutil.copyfile(path_orig, path_new)

        if len(image_names) == 0:
            return ''
        else:
            return figure_rst(image_names, gallery_conf['src_dir'])


def _find_images(path, image_extensions=['jpg', 'jpeg', 'png', 'gif']):
    """Find all unique image paths for a set of extensions."""
    image_files = set()
    for ext in image_extensions:
        this_ext_files = set(glob.glob(os.path.join(path, '*.'+ext)))
        image_files = image_files.union(this_ext_files)
    return image_files
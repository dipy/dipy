# init for sext package
""" Setuptools extensions that can be shared across projects

Typical use for these routines is as a git subtree merge

For example::

    # Add a remote pointing to repository
    git remote add nisext git://github.com/nipy/nisext.git
    git fetch nisext
    # Label nisext history as merged
    git merge -s ours --no-commit nisext/master
    # Read nisext contents as nisext subdirectory
    git read-tree --prefix=nisext/ -u nisext/master
    git commit -m "Merge nisext project as subtree"

Then you would typically add a makefile target like::

    # Update nisext subtree from remote
    update-nisext:
            git fetch nisext
            git merge --squash -s subtree --no-commit nisext/master

and commit when you have changes you want. This allows you to keep the nisext
tree updated from the upstream repository, but the tree will be there and ready
for someone without this machinery or remote.
"""


.. _python3:

############################################
Keeping code compatible with Pythons 2 and 3
############################################

DIPY 0.16 (and lower) supports Python versions from 2.6 to 3.5. In order to maintain code that supports both
Python 2 and Python 3 versions, please follow these instructions.

There is useful advice here:

* http://docs.python.org/3/howto/pyporting.html
* http://python3porting.com/differences.html
* http://ptgmedia.pearsoncmg.com/imprint_downloads/informit/promotions/python/python2python3.pdf

**************
Future imports
**************

For any modules with print statements, and for any modules where you remember,
please put::

    from __future__ import division, print_function, absolute_import

As the first code line of the file, to use Python 3 behavior by default.

*****
Print
*****

In Python 3, ``print`` is a function. Please use the ``__future__`` import above,
and the function form:``print(something)``, whenever ``print`` is used.

********
Division
********

In Python 2, integer division returns integers, while in Python 3 ``3/2``
returns ``1.5`` not ``1``.  It's very good to remember to put the ``__future__``
import above at the top of the file to make this default everywhere.

*************
Moved modules
*************

There are compatibility routines in :mod:`dipy.utils.six`.  You can often get
modules that have moved between the versions with (e.g.)::

    from dipy.utils.six.moves import configparser

See the ``six.py`` code and `the six.py docs <http://pythonhosted.org/six>`_.

*************
Range, xrange
*************

``range`` returns an iterator in Python3, and ``xrange`` is therefore redundant,
and it has been removed.  Get ``xrange`` for Python 2, ``range`` for Python 3
with::

    from dipy.utils.six.moves import xrange

Or you might want to stick to ``range`` for Python 2 and Python 3, especially
for small lists where the memory benefit for ``xrange`` is small.

Because ``range`` returns an iterator for Python 3, you may need to wrap some
calls to range with ``list(range(N))`` to make the code compatible with Python 2
and Python 3.

******
Reduce
******

Python 3 removed ``reduce`` from the builtin namespace, this import works for
both Python 2 and Python 3::

    from functools import reduce

*******
Strings
*******

The major difference between Python 2 and Python 3 is the string handling.
Strings (``str``) are always unicode, and so::

    my_str = 'A string'

in Python 3 will result in a unicode string.  You also need to be much more
explicit when opening files; If you want bytes, use: ``open(fname, "rb")``. If
you want unicode: ``open(fname, "rt")``.  In the same way you need to be explicit if
you want ``import io; io.StringIO`` or ``io.BytesIO`` for your file-like objects
containing strings or bytes.

``basestring`` has been removed in Python 3.  To test whether something is a
string, use::

   from dipy.utils.six import string_types

   isinstance(a_variable, string_types)

*************
Next function
*************

In versions of Python from 2.6 and on there is a function ``next`` in the
builtin namespace, that returns the next result from an iterable thing.   In
Python 3, meanwhile, the ``.next()`` method on generators has gone, replaced by
``.__next__()``.  So, prefer ``next(obj)`` to ``obj.next()`` for generators, and
in general when getting the next thing from an iterable.

******
Except
******

You can't get away with ``except ValueError, err`` now, because that raises a
syntax error for Python 3.  Use ``except ValueError as err`` instead.

************
Dictionaries
************

You've lost ``d.has_key("hello")`` for dictionaries, use ``"hello" in d``
instead.

``d.items()`` returns an iterator.  If you need a list, use ``list(d.items()``.
``d.iteritems()`` was removed in Python 3 because it is redundant. Use
``d.items()`` instead.

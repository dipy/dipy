"""
This was taken directly from the file test_docscrape.py of numpydoc package.

Copyright (C) 2008 Stefan van der Walt <stefan@mentat.za.net>,
Pauli Virtanen <pav@iki.fi>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

# -*- encoding:utf-8 -*-

import textwrap

from dipy.workflows.docstring_parser import NumpyDocString
import numpy.testing as npt


doc_txt = """\
  numpy.multivariate_normal(mean, cov, shape=None, spam=None)

  Draw values from a multivariate normal distribution with specified
  mean and covariance.

  The multivariate normal or Gaussian distribution is a generalisation
  of the one-dimensional normal distribution to higher dimensions.

  Parameters
  ----------
  mean : (N,) ndarray
      Mean of the N-dimensional distribution.

      .. math::

         (1+2+3)/3

  cov : (N, N) ndarray
      Covariance matrix of the distribution.
  shape : tuple of ints
      Given a shape of, for example, (m,n,k), m*n*k samples are
      generated, and packed in an m-by-n-by-k arrangement.  Because
      each sample is N-dimensional, the output shape is (m,n,k,N).

  Returns
  -------
  out : ndarray
      The drawn samples, arranged according to `shape`.  If the
      shape given is (m,n,...), then the shape of `out` is is
      (m,n,...,N).

      In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
      value drawn from the distribution.
  list of str
      This is not a real return value.  It exists to test
      anonymous return values.

  Other Parameters
  ----------------
  spam : parrot
      A parrot off its mortal coil.

  Raises
  ------
  RuntimeError
      Some error

  Warns
  -----
  RuntimeWarning
      Some warning

  Warnings
  --------
  Certain warnings apply.

  Notes
  -----
  Instead of specifying the full covariance matrix, popular
  approximations include:

    - Spherical covariance (`cov` is a multiple of the identity matrix)
    - Diagonal covariance(`cov` has non-negative elements only on the diagonal)

  This geometrical property can be seen in two dimensions by plotting
  generated data-points:

  >>> mean = [0,0]
  >>> cov = [[1,0],[0,100]] # diagonal covariance, points lie on x or y-axis

  >>> x,y = multivariate_normal(mean,cov,5000).T
  >>> plt.plot(x,y,'x'); plt.axis('equal'); plt.show()

  Note that the covariance matrix must be symmetric and non-negative
  definite.

  References
  ----------
  .. [1] A. Papoulis, "Probability, Random Variables, and Stochastic
         Processes," 3rd ed., McGraw-Hill Companies, 1991
  .. [2] R.O. Duda, P.E. Hart, and D.G. Stork, "Pattern Classification,"
         2nd ed., Wiley, 2001.

  See Also
  --------
  some, other, funcs
  otherfunc : relationship

  Examples
  --------
  >>> mean = (1,2)
  >>> cov = [[1,0],[1,0]]
  >>> x = multivariate_normal(mean,cov,(3,3))
  >>> print x.shape
  (3, 3, 2)

  The following is probably true, given that 0.6 is roughly twice the
  standard deviation:

  >>> print list( (x[0,0,:] - mean) < 0.6 )
  [True, True]

  .. index:: random
     :refguide: random;distributions, random;gauss

  """
doc = NumpyDocString(doc_txt)

doc_yields_txt = """
Test generator

Yields
------
a : int
    The number of apples.
b : int
    The number of bananas.
int
    The number of unknowns.
"""
doc_yields = NumpyDocString(doc_yields_txt)


def test_signature():
    npt.assert_(doc['Signature'].startswith('numpy.multivariate_normal('))
    npt.assert_(doc['Signature'].endswith('spam=None)'))


def test_summary():
    npt.assert_(doc['Summary'][0].startswith('Draw values'))
    npt.assert_(doc['Summary'][-1].endswith('covariance.'))


def test_extended_summary():
    npt.assert_(doc['Extended Summary'][0].
                startswith('The multivariate normal'))


def test_parameters():
    npt.assert_equal(len(doc['Parameters']), 3)
    npt.assert_equal(
        [n for n, _, _ in doc['Parameters']], ['mean', 'cov', 'shape'])

    arg, arg_type, desc = doc['Parameters'][1]
    npt.assert_equal(arg_type, '(N, N) ndarray')
    npt.assert_(desc[0].startswith('Covariance matrix'))
    npt.assert_equal(doc['Parameters'][0][-1][-2], '   (1+2+3)/3')


def test_other_parameters():
    npt.assert_equal(len(doc['Other Parameters']), 1)
    npt.assert_equal([n for n, _, _ in doc['Other Parameters']], ['spam'])
    arg, arg_type, desc = doc['Other Parameters'][0]
    npt.assert_equal(arg_type, 'parrot')
    npt.assert_(desc[0].startswith('A parrot off its mortal coil'))


def test_returns():
    npt.assert_equal(len(doc['Returns']), 2)
    arg, arg_type, desc = doc['Returns'][0]
    npt.assert_equal(arg, 'out')
    npt.assert_equal(arg_type, 'ndarray')
    npt.assert_(desc[0].startswith('The drawn samples'))
    npt.assert_(desc[-1].endswith('distribution.'))

    arg, arg_type, desc = doc['Returns'][1]
    npt.assert_equal(arg, 'list of str')
    npt.assert_equal(arg_type, '')
    npt.assert_(desc[0].startswith('This is not a real'))
    npt.assert_(desc[-1].endswith('anonymous return values.'))


def test_notes():
    npt.assert_(doc['Notes'][0].startswith('Instead'))
    npt.assert_(doc['Notes'][-1].endswith('definite.'))
    npt.assert_equal(len(doc['Notes']), 17)


def test_references():
    npt.assert_(doc['References'][0].startswith('..'))
    npt.assert_(doc['References'][-1].endswith('2001.'))


def test_examples():
    npt.assert_(doc['Examples'][0].startswith('>>>'))
    npt.assert_(doc['Examples'][-1].endswith('True]'))


def test_index():
    npt.assert_equal(doc['index']['default'], 'random')
    npt.assert_equal(len(doc['index']), 2)
    npt.assert_equal(len(doc['index']['refguide']), 2)


def non_blank_line_by_line_compare(a, b):
    a = textwrap.dedent(a)
    b = textwrap.dedent(b)
    a = [l.rstrip() for l in a.split('\n') if l.strip()]
    b = [l.rstrip() for l in b.split('\n') if l.strip()]
    for n, line in enumerate(a):
        if not line == b[n]:
            raise AssertionError("Lines %s of a and b differ: "
                                 "\n>>> %s\n<<< %s\n" %
                                 (n, line, b[n]))


def test_str():
    # doc_txt has the order of Notes and See Also sections flipped.
    # This should be handled automatically, and so, one thing this test does
    # is to make sure that See Also precedes Notes in the output.
    non_blank_line_by_line_compare(str(doc),
                                   """numpy.multivariate_normal(mean, cov, shape=None, spam=None)

Draw values from a multivariate normal distribution with specified
mean and covariance.

The multivariate normal or Gaussian distribution is a generalisation
of the one-dimensional normal distribution to higher dimensions.

Parameters
----------
mean : (N,) ndarray
    Mean of the N-dimensional distribution.

    .. math::

       (1+2+3)/3

cov : (N, N) ndarray
    Covariance matrix of the distribution.
shape : tuple of ints
    Given a shape of, for example, (m,n,k), m*n*k samples are
    generated, and packed in an m-by-n-by-k arrangement.  Because
    each sample is N-dimensional, the output shape is (m,n,k,N).

Returns
-------
out : ndarray
    The drawn samples, arranged according to `shape`.  If the
    shape given is (m,n,...), then the shape of `out` is is
    (m,n,...,N).

    In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
    value drawn from the distribution.
list of str
    This is not a real return value.  It exists to test
    anonymous return values.

Other Parameters
----------------
spam : parrot
    A parrot off its mortal coil.

Raises
------
RuntimeError
    Some error

Warns
-----
RuntimeWarning
    Some warning

Warnings
--------
Certain warnings apply.

See Also
--------
`some`_, `other`_, `funcs`_

`otherfunc`_
    relationship

Notes
-----
Instead of specifying the full covariance matrix, popular
approximations include:

  - Spherical covariance (`cov` is a multiple of the identity matrix)
  - Diagonal covariance(`cov` has non-negative elements only on the diagonal)

This geometrical property can be seen in two dimensions by plotting
generated data-points:

>>> mean = [0,0]
>>> cov = [[1,0],[0,100]] # diagonal covariance, points lie on x or y-axis

>>> x,y = multivariate_normal(mean,cov,5000).T
>>> plt.plot(x,y,'x'); plt.axis('equal'); plt.show()

Note that the covariance matrix must be symmetric and non-negative
definite.

References
----------
.. [1] A. Papoulis, "Probability, Random Variables, and Stochastic
       Processes," 3rd ed., McGraw-Hill Companies, 1991
.. [2] R.O. Duda, P.E. Hart, and D.G. Stork, "Pattern Classification,"
       2nd ed., Wiley, 2001.

Examples
--------
>>> mean = (1,2)
>>> cov = [[1,0],[1,0]]
>>> x = multivariate_normal(mean,cov,(3,3))
>>> print x.shape
(3, 3, 2)

The following is probably true, given that 0.6 is roughly twice the
standard deviation:

>>> print list( (x[0,0,:] - mean) < 0.6 )
[True, True]

.. index:: random
   :refguide: random;distributions, random;gauss""")


doc2 = NumpyDocString("""
    Returns array of indices of the maximum values of along the given axis.

    Parameters
    ----------
    a : {array_like}
        Array to look in.
    axis : {None, integer}
        If None, the index is into the flattened array, otherwise along
        the specified axis""")


def test_parameters_without_extended_description():
    npt.assert_equal(len(doc2['Parameters']), 2)

doc3 = NumpyDocString("""
    my_signature(*params, **kwds)

    Return this and that.
    """)

doc5 = NumpyDocString(
    """
    a.something()

    Raises
    ------
    LinAlgException
        If array is singular.

    Warns
    -----
    SomeWarning
        If needed
    """)


def test_raises():
    npt.assert_equal(len(doc5['Raises']), 1)
    name, _, desc = doc5['Raises'][0]
    npt.assert_equal(name, 'LinAlgException')
    npt.assert_equal(desc, ['If array is singular.'])


def test_warns():
    npt.assert_equal(len(doc5['Warns']), 1)
    name, _, desc = doc5['Warns'][0]
    npt.assert_equal(name, 'SomeWarning')
    npt.assert_equal(desc, ['If needed'])


def test_see_also():
    doc6 = NumpyDocString(
        """
    z(x,theta)

    See Also
    --------
    func_a, func_b, func_c
    func_d : some equivalent func
    foo.func_e : some other func over
             multiple lines
    func_f, func_g, :meth:`func_h`, func_j,
    func_k
    :obj:`baz.obj_q`
    :class:`class_j`: fubar
        foobar
    """)

    npt.assert_equal(len(doc6['See Also']), 12)
    for func, desc, role in doc6['See Also']:
        if func in ('func_a', 'func_b', 'func_c', 'func_f',
                    'func_g', 'func_h', 'func_j', 'func_k', 'baz.obj_q'):
            assert not desc
        else:
            assert desc

        if func == 'func_h':
            assert role == 'meth'
        elif func == 'baz.obj_q':
            assert role == 'obj'
        elif func == 'class_j':
            assert role == 'class'
        else:
            assert role is None

        if func == 'func_d':
            assert desc == ['some equivalent func']
        elif func == 'foo.func_e':
            assert desc == ['some other func over', 'multiple lines']
        elif func == 'class_j':
            assert desc == ['fubar', 'foobar']

doc7 = NumpyDocString("""

        Doc starts on second line.

        """)


def test_empty_first_line():
    assert doc7['Summary'][0].startswith('Doc starts')


def test_duplicate_signature():
    # Duplicate function signatures occur e.g. in ufuncs, when the
    # automatic mechanism adds one, and a more detailed comes from the
    # docstring itself.

    doc = NumpyDocString(
        """
    z(x1, x2)

    z(a, theta)
    """)

    assert doc['Signature'].strip() == 'z(a, theta)'


class_doc_txt = """
    Foo

    Parameters
    ----------
    f : callable ``f(t, y, *f_args)``
        Aaa.
    jac : callable ``jac(t, y, *jac_args)``
        Bbb.

    Attributes
    ----------
    t : float
        Current time.
    y : ndarray
        Current variable values.
    x : float
        Some parameter

    Methods
    -------
    a
    b
    c

    Examples
    --------
    For usage examples, see `ode`.
"""

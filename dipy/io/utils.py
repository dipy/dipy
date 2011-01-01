''' Utility functions for file formats '''

import sys

import numpy as np

sys_is_le = sys.byteorder == 'little'
native_code = sys_is_le and '<' or '>'
swapped_code = sys_is_le and '>' or '<'

default_compresslevel = 1

endian_codes = (# numpy code, aliases
    ('<', 'little', 'l', 'le', 'L', 'LE'),
    ('>', 'big', 'BIG', 'b', 'be', 'B', 'BE'),
    (native_code, 'native', 'n', 'N', '=', '|', 'i', 'I'),
    (swapped_code, 'swapped', 's', 'S', '!'))
# We'll put these into the Recoder class after we define it


class Recoder(object):
    ''' class to return canonical code(s) from code or aliases

    The concept is a lot easier to read in the implementation and
    tests than it is to explain, so...

    >>> # If you have some codes, and several aliases, like this:
    >>> code1 = 1; aliases1=['one', 'first']
    >>> code2 = 2; aliases2=['two', 'second']
    >>> # You might want to do this:
    >>> codes = [[code1]+aliases1,[code2]+aliases2]
    >>> recodes = Recoder(codes)
    >>> recodes.code['one']
    1
    >>> recodes.code['second']
    2
    >>> recodes.code[2]
    2
    >>> # Or maybe you have a code, a label and some aliases
    >>> codes=((1,'label1','one', 'first'),(2,'label2','two'))
    >>> # you might want to get back the code or the label
    >>> recodes = Recoder(codes, fields=('code','label'))
    >>> recodes.code['first']
    1
    >>> recodes.code['label1']
    1
    >>> recodes.label[2]
    'label2'
    >>> # For convenience, you can get the first entered name by
    >>> # indexing the object directly
    >>> recodes[2]
    2
    '''
    def __init__(self, codes, fields=('code',)):
        ''' Create recoder object

	``codes`` give a sequence of code, alias sequences
	``fields`` are names by which the entries in these sequences can be
	accessed.

	By default ``fields`` gives the first column the name
	"code".  The first column is the vector of first entries
	in each of the sequences found in ``codes``.  Thence you can
	get the equivalent first column value with ob.code[value],
	where value can be a first column value, or a value in any of
	the other columns in that sequence. 

	You can give other columns names too, and access them in the
	same way - see the examples in the class docstring. 

        Parameters
        ------------
        codes : seqence of sequences
            Each sequence defines values (codes) that are equivalent
        fields : {('code',) string sequence}, optional
            names by which elements in sequences can be accesssed

        '''
        self.fields = fields
        self.field1 = {} # a placeholder for the check below
        for name in fields:
            if name in self.__dict__:
                raise KeyError('Input name %s already in object dict'
                               % name)
            self.__dict__[name] = {}
        self.field1 = self.__dict__[fields[0]]
        self.add_codes(codes)
        
    def add_codes(self, codes):
        ''' Add codes to object

	    >>> codes = ((1, 'one'), (2, 'two'))
	    >>> rc = Recoder(codes)
        >>> rc.value_set() == set((1,2))
        True
        >>> rc.add_codes(((3, 'three'), (1, 'first')))
        >>> rc.value_set() == set((1,2,3))
        True
        '''
        for vals in codes:
            for val in vals:
                for ind, name in enumerate(self.fields):
                    self.__dict__[name][val] = vals[ind]
        
        
    def __getitem__(self, key):
        ''' Return value from field1 dictionary (first column of values)

	    Returns same value as ``obj.field1[key]`` and, with the
        default initializing ``fields`` argument of fields=('code',),
        this will return the same as ``obj.code[key]``

	    >>> codes = ((1, 'one'), (2, 'two'))
	    >>> Recoder(codes)['two']
	    2
        '''
        return self.field1[key]

    def keys(self):
    	''' Return all available code and alias values 

	    Returns same value as ``obj.field1.keys()`` and, with the
        default initializing ``fields`` argument of fields=('code',),
        this will return the same as ``obj.code.keys()``

	    >>> codes = ((1, 'one'), (2, 'two'), (1, 'repeat value'))
	    >>> k = Recoder(codes).keys()
	    >>> k.sort() # Just to guarantee order for doctest output
	    >>> k
	    [1, 2, 'one', 'repeat value', 'two']
	    '''
        return self.field1.keys()

    def value_set(self, name=None):
        ''' Return set of possible returned values for column

        By default, the column is the first column.

	    Returns same values as ``set(obj.field1.values())`` and,
        with the default initializing ``fields`` argument of
        fields=('code',), this will return the same as
        ``set(obj.code.values())``

        Parameters
        ------------
        name : {None, string}
            Where default of None gives result for first column

        Returns
        ---------
        val_set : set
           set of all values for `name`

        Examples
        -----------
        >>> codes = ((1, 'one'), (2, 'two'), (1, 'repeat value'))
        >>> vs = Recoder(codes).value_set()
        >>> vs == set([1, 2]) # Sets are not ordered, hence this test
        True
        >>> rc = Recoder(codes, fields=('code', 'label'))
        >>> rc.value_set('label') == set(('one', 'two', 'repeat value'))
        True
        
        '''
        if name is None:
            d = self.field1
        else:
            d = self.__dict__[name]
        return set(d.values())

    
# Endian code aliases
endian_codes = Recoder(endian_codes)


def allopen(fname, *args, **kwargs):
    ''' Generic file-like object open

    If input ``fname`` already looks like a file, pass through.
    If ``fname`` ends with recognizable compressed types, use python
    libraries to open as file-like objects (read or write)
    Otherwise, use standard ``open``.
    '''
    if hasattr(fname, 'write'):
        return fname
    if args:
        mode = args[0]
    elif 'mode' in kwargs:
        mode = kwargs['mode']
    else:
        mode = 'rb'
    if fname.endswith('.gz'):
        if ('w' in mode and
            len(args) < 2 and
            not 'compresslevel' in kwargs):
            kwargs['compresslevel'] = default_compresslevel
        import gzip
        opener = gzip.open
    elif fname.endswith('.bz2'):
        if ('w' in mode and
            len(args) < 3 and
            not 'compresslevel' in kwargs):
            kwargs['compresslevel'] = default_compresslevel
        import bz2
        opener = bz2.BZ2File
    else:
        opener = open
    return opener(fname, *args, **kwargs)

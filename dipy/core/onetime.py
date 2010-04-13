"""Descriptor support for NIPY.

Utilities to support special Python descriptors [1,2], in particular the use of
a useful pattern for properties we call 'one time properties'.  These are
object attributes which are declared as properties, but become regular
attributes once they've been read the first time.  They can thus be evaluated
later in the object's life cycle, but once evaluated they become normal, static
attributes with no function call overhead on access or any other constraints.

References
----------
[1] How-To Guide for Descriptors, Raymond
Hettinger. http://users.rcn.com/python/download/Descriptor.htm

[2] Python data model, http://docs.python.org/reference/datamodel.html
"""

class OneTimeProperty(object):
   """A descriptor to make special properties that become normal attributes.
   """
   def __init__(self,func):
       """Create a OneTimeProperty instance.

        Parameters
        ----------
          func : method
          
            The method that will be called the first time to compute a value.
            Afterwards, the method's name will be a standard attribute holding
            the value of this computation.
            """
       self.getter = func
       self.name = func.func_name

   def __get__(self,obj,type=None):
       """This will be called on attribute access on the class or instance. """

       if obj is None:
           # Being called on the class, return the original function. This way,
           # introspection works on the class.
           return self.getter

       val = self.getter(obj)
       #print "** setattr_on_read - loading '%s'" % self.name  # dbg
       setattr(obj, self.name, val)
       return val


def setattr_on_read(func):
# XXX - beetter names for this?
# - cor_property (copy on read property)
# - sor_property (set on read property)
# - prop2attr_on_read
#... ?

    """Decorator to create OneTimeProperty attributes.

    Parameters
    ----------
      func : method
        The method that will be called the first time to compute a value.
        Afterwards, the method's name will be a standard attribute holding the
        value of this computation.

    Examples
    --------
    >>> class MagicProp(object):
    ...     @setattr_on_read
    ...     def a(self):
    ...         return 99
    ...     
    >>> x = MagicProp()
    >>> 'a' in x.__dict__
    False
    >>> x.a
    99
    >>> 'a' in x.__dict__
    True
    """
    return OneTimeProperty(func)

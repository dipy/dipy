""" Classes to provide and API like traits, even if traits is absent """

from ..utils.optpkg import optional_package

# Import traits as optional package
tapi, have_traits, setup_module = optional_package('traits.api')

# Import traitsui as optional package
tuapi, _, _ = optional_package('traitsui.api')

if not have_traits:
    class FlexiTrip(object):
        """ Class for which instances raise error on attribute access

        Like ``tripwire.TripWire`` but allowing any init arguments
        """
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, attr_name):
            raise RuntimeError("You need 'traits' for this object to work")

    def decotrip(func):
        ''' Decorator to return tripping instance '''
        return FlexiTrip

    tapi.HasTraits = FlexiTrip
    tapi.HasStrictTraits = FlexiTrip
    tapi.File = FlexiTrip
    tapi.String = FlexiTrip
    tapi.Float = FlexiTrip
    tapi.Int = FlexiTrip
    tapi.Bool = FlexiTrip
    tapi.List = FlexiTrip
    tapi.Tuple = FlexiTrip
    tapi.Range = FlexiTrip
    tapi.Array = FlexiTrip
    tapi.DelegatesTo = FlexiTrip
    tapi.Instance = FlexiTrip
    tapi.Enum = FlexiTrip
    tapi.on_trait_change = decotrip
    # UI
    tuapi.Item = FlexiTrip
    tuapi.Group = FlexiTrip
    tuapi.View = FlexiTrip
    tuapi.ArrayEditor = FlexiTrip

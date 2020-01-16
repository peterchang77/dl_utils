def overload(default):
    """
    Decorator function for loading default class method / attrs

    Example:

      @overload(Client)
      def preprocess(self, arrays, **kwargs):
        ...

    This snippet will overload the default Client.preprocess(...) method defined in the
    original class code.

    """
    def wrapper(func):
        setattr(default, func.__name__, func)

    return wrapper 


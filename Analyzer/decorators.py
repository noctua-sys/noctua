import functools
import Analyzer.notify


def returns(value):
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            if Analyzer.notify.get_current_path():
                return value
            else:
                return f(*args, **kwargs)
        return decorated
    return decorator



def returns_by(thunk):
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            if Analyzer.notify.get_current_path():
                return thunk()
            else:
                return f(*args, **kwargs)
        return decorated
    return decorator



_current_site = None

from Coord.client import coord
import time
from rest_framework import mixins

def current_site():
    if _current_site:
        return _current_site
    else:
        raise RuntimeError('you did not specify the site number')

def wrap(op_name, method, request, *args, **kwargs):
    while True:
        time.sleep(0.001)
        reqid = coord.add(current_site(), op_name)
        if reqid: break
        # else: time.sleep(0.00001)
    try:
        ret = method(request, *args, **kwargs)
        coord.remove(reqid)
        return ret
    except Exception as e:
        coord.remove(reqid)
        raise e

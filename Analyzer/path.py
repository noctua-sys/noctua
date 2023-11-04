import copy
from collections import OrderedDict
from typing import Dict
from Verifier.checker import *


__all__ = ["Path"]


L = 'L'
R = 'R'
Comparision = ['>', '>=', '==', '<', '<=']


class Path:
    """
    Maintain the information about a program path.
    """
    def __init__(self):
        self.state = OrderedDict()
        self.free_vars = dict()   # VarName -> VarType (str)
        self.path_conds = set()   # Set[Expr]
        self.effects = list()     # List[Command]
        self.unique_ids = dict()   # Dict[str,str]
        self.bound = 1        # how many times the same value can be returned for an object
        self.counter = dict() # counter records the number of times the current value is returned.
        self.ir = ""

        self.num_insert = 0   # How many inserts happened.

    def __str__(self):
        return f'[{len(self.free_vars)} args, {len(self.path_conds)} conds, {len(self.effects)} effects]'

    def make_immutable(self):
        """
        Get an immutable summary of a complete path.

        The return value cannot be used as an active path.
        """
        ret = Path()
        ret.free_vars = copy.copy(self.free_vars)
        ret.unique_ids = copy.copy(self.unique_ids)
        ret.path_conds = copy.copy(self.path_conds)
        ret.effects = copy.copy(self.effects)
        ret.bound = self.bound
        ret.counter = copy.copy(self.counter)
        ret.num_insert = self.num_insert
        return ret

    def is_effectful(self):
        return len(self.effects) > 0

    def add_free_var(self, var_name: str, type=None):
        # assert var_name not in self.free_vars
        assert isinstance(var_name, str)
        # TODO(ksqsf): add type probing
        # TODO(ksqsf): what if the same var is added many times?

        # FIXME: APP-SPECIFIC.
        if type is None:
            if var_name.endswith('text'):
                type = Type.STR()
            elif var_name.endswith('anonymous'):
                type = Type.BOOL()
            else:
                type = Type.INT()

        assert isinstance(type, Type)
        self.free_vars[var_name] = type

    def add_path_cond(self, expr):
        from Verifier.checker import Expr
        assert isinstance(expr, Expr)
        self.path_conds.add(expr)
        # self.path_conds.add(_make_immutable(expr))

    def add_effect(self, cmd):
        from Verifier.checker import Command
        assert isinstance(cmd, Command)
        self.effects.append(cmd)

    def set_return_value(self, retval):
        self.retval = retval

    def has_more_paths(self):
        return len(self.state) > 0

    def determine_bool(self, id):
        """
        Determine the boolean value of node `id` at this particular point.
        If `id` is discovered for the first time, it is inserted into the path state.
        """
        id = _make_immutable(id)
        if id not in self.state:
            #print('[DBG] Found a new bool node %s, giving False' % str(id))
            self.state[id] = L
            self.counter[id] = 1
            return False
        elif self.state[id] == L:
            #print('[DBG] Bool node %s is False' % str(id))
            self.counter.setdefault(id, 0)
            self.counter[id] += 1
            # TODO(ksqsf): cut the remaining part off self.state if cnt > bound
            return False
        elif self.state[id] == R:
            #print('[DBG] Bool node %s is True' % str(id))
            self.counter.setdefault(id, 0)
            self.counter[id] += 1
            # TODO(ksqsf): the same as above
            # TODO(ksqsf): How to handle typical loop conditions like True -> True -> False ?
            return True
        else:
            # Unreachable
            assert False

    def advance(self):
        """
        Advance explore_state after a complete path is discovered.
        """
        self.clear()
        while len(self.state) > 0:
            (key, value) = self.state.popitem()
            if value == L:
                self.state[key] = R
                self.counter[key] = 0
                return

    def clear(self):
        self.free_vars.clear()
        self.path_conds.clear()
        self.effects.clear()
        self.counter.clear()
        self.unique_ids.clear()
        self.num_insert = 0

    def fresh_insert_obj_name(self):
        self.num_insert += 1
        return f'obj{self.num_insert}'

    def add_unique_id(self, id_name: str, mname: str):
        assert isinstance(id_name, str)
        self.unique_ids[id_name] = mname


def _make_immutable(obj):
    # Break circular import
    from Analyzer.symbolic import Sym

    if isinstance(obj, Sym):
        return _make_immutable(obj.expr)
    elif obj is None:
        return None
    elif isinstance(obj, (int, bool, str)):
        return obj
    elif isinstance(obj, (set, tuple, list)):
        return tuple(_make_immutable(elt) for elt in obj)
    elif isinstance(obj, dict):
        keys = [_make_immutable(k) for k in sorted(list(obj.keys()))]
        values = [_make_immutable(obj[k]) for k in keys]
        print ('[DBG] values', values)
        return tuple(zip(keys, values))
    elif isinstance(obj, (Free)):
        return obj.name
    elif isinstance(obj, FreeObj):
        return obj.name
    elif isinstance(obj, Expr):
        return obj.cached_pprint(0)
    else:
        raise ValueError('Cannot make %s immutable' % (type(obj)))

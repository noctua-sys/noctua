import bdb
import linecache
from typing import Callable, Dict, Generic, TypeVar, Union
import traceback

from Analyzer.path import Path
from Analyzer.notify import *
from Analyzer import info
from Analyzer.symbolic import Sym


Ret = TypeVar('Ret')


class PathFinder(bdb.Bdb, Generic[Ret]):
    def __init__(self, func: Callable[[], Ret], extra_args, extra_kwargs: Dict[str, Sym]):
        super().__init__(skip=['django*', 'rest_framework.*', 'Analyzer.*', 'Verifier.*', 'z3*', 'adt*'] + info.python_std_modules)
        self.func = func
        self.extra_args = extra_args
        self.extra_kwargs = extra_kwargs
        self._debug = True

        # Internal states
        self.cur_path = Path()
        self.paths = []
        self.number_of_explored_paths = 0

    def one_path(self, record=True) -> Union[None, Ret]:
        # self.cur_path does not change (for a specific view
        # function).  When a path is analyzed, invoke
        # cur_path.make_immutable() to obtain an immutable copy.
        set_current_path(self.cur_path)
        retval = None

        try:
            self.before_invocation()
            retval = self.runcall(self.func, *self.extra_args, **self.extra_kwargs)
            self.cur_path.set_return_value(retval)
            self.log('Found a new path with retval = %s' % repr(retval)[:500])
            if record and self.cur_path.is_effectful():
                self.paths.append(self.cur_path.make_immutable())
                # self.debug('argument =', self.cur_path.free_vars)
                # self.debug('condition =', self.cur_path.path_conds)
                # self.debug('effect =', self.cur_path.effects)
        except:
            self.error('This path raised an exception:', traceback.format_exc())

        self.cur_path.advance()
        self.number_of_explored_paths += 1
        return retval

    def before_invocation(self):
        """Add the query parameters to the list of free variables."""
        for sym in self.extra_kwargs.values():
            assert isinstance(sym.expr, Free)
            free = sym.expr
            self.cur_path.add_free_var(free.name, free.typ)

    def explore(self):
        self.one_path()
        while self.cur_path.has_more_paths():
            self.log('Continuing...')
            self.one_path()

        self.debug('Number of effectful paths discovered = %d' % (len(self.paths)))
        for i, path in enumerate(self.paths):
            self.debug('Path %d = %s' % (i + 1, str(path)))

    def error(self, *args):
        message = ' '.join(str(arg) for arg in args)
        print('[ERR]', message)

    def debug(self, *args):
        if not self._debug:
            return
        message = ' '.join(str(arg) for arg in args)
        print('[DBG]', message)

    def log(self, *args):
        if not self._debug:
            return
        message = ' '.join(str(arg) for arg in args)
        print('+++', message)

    #############
    # BDB hooks #
    #############
    def user_line(self, frame):
        name = frame.f_code.co_name
        file = self.canonic(frame.f_code.co_filename)
        line = linecache.getline(file, frame.f_lineno, frame.f_globals)
        self.log(file, frame.f_lineno, name, ':', line.strip())

    def user_exception(self, frame, exc_info):
        self.log('exception', exc_info)
        print(''.join(traceback.format_exception(*exc_info)))
        pass

    def user_call(self, frame, argument_list):
        self.log('call', frame.f_code.co_name, argument_list)
        pass

    def user_return(self, frame, return_value):
        self.log('return', str(return_value)[:500])
        pass

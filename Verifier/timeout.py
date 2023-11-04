import os
import pickle
import select
import signal
import math
import traceback
from typing import TypeVar, Callable


POLL_TIMEOUT = 0.05


RetType = TypeVar('RetType')


class WorkerProcessError(Exception):
    pass


def timedcall(timeout: float, func: Callable[..., RetType], *args, **kwargs) -> RetType:
    """
    Fork and run @func in the child process using up to @timeout seconds.

    If @func does not return in @timeout seconds, the process is
    killed, and `TimeoutError` is raised.

    If @func raises any exception, it is wrapped in
    `WorkerProcessError` and raised.
    """
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            ret = func(*args, **kwargs)
            data = pickle.dumps(('return', ret))
        except Exception as e:
            traceback.print_exception(e)
            data = pickle.dumps(('raise', e))
        os.write(w, data)
        os.close(w)
        os._exit(0)
    else:
        os.close(w)
        exited = False
        interrupted = False
        try:
            for _ in range(math.ceil(timeout / POLL_TIMEOUT)):
                # Poll the fd.  Do not rely on signals.
                rs, _, _ = select.select([r], [], [], POLL_TIMEOUT)
                if len(rs) == 0:
                    continue
                else:
                    exited = True
                    break
        except KeyboardInterrupt:
            interrupted = True
        # The worker has either timed out, or exited; or the waiter is interrupted.
        if not exited or interrupted:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            os.close(r)
            if interrupted:
                raise KeyboardInterrupt()
            else:
                raise TimeoutError('timedcall')
        else:
            os.waitpid(pid, 0)
            buf = bytearray()
            while True:
                cur = os.read(r, 4096)
                if len(cur) == 0:
                    break
                buf += cur
            os.close(r)
            kind, data = pickle.loads(buf)
            if kind == 'return':
                return data
            elif kind == 'raise':
                raise WorkerProcessError() from data
            else:
                raise ValueError(f'unknown result kind "{kind}" from worker')
                    

'''
from datetime import datetime

def test():
    return 1

start = datetime.now()
try:
    ret = timedcall(2.0, test)
    print(f'Finished in {datetime.now() - start}! Result =', ret)
except TimeoutError:
    print(f'Timeout in {datetime.now() - start}')
'''

import signal
from datetime import datetime, timedelta
import sys


TIMEOUT_DEBUG = True


class TimeoutManager:
    def __init__(self, timeout: int):
        self.timeout = timeout
        self.elapsed = timedelta(0)

    def __enter__(self):
        def _handle_timeout(signum, frame):
            global TIMEOUT_DEBUG
            if TIMEOUT_DEBUG:
                print(f'received SIGALRM at {datetime.now()}')
                sys.stdout.flush()
            raise TimeoutError('Time limit reached')

        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(self.timeout)

        self.start = datetime.now()

        return self

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

        self.end = datetime.now()
        self.elapsed = self.end - self.start
        global TIMEOUT_DEBUG
        if TIMEOUT_DEBUG:
            print('elapsed: {}'.format(self.elapsed))
            sys.stdout.flush()

import datetime


class TimeCounter:
    def __init__(self):
        self.started = dict()
        self.ended = dict()
        self.elapsed: dict[object, datetime.timedelta] = dict()
        self.hier_sum = dict()
        self.hier_cnt = dict()

    def time(self, id):
        class TimeCounterMgr:
            def __init__(self, id, counter):
                self.id = id
                self.counter = counter
            def __enter__(self):
                self.counter.start(self.id)
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.counter.stop(self.id)
        return TimeCounterMgr(id, self)

    def start(self, id):
        self.started[id] = datetime.datetime.now()

    def stop(self, id):
        self.ended[id] = datetime.datetime.now()
        self.elapsed[id] = self.ended[id] - self.started[id]
        if isinstance(id, tuple):
            id = list(id)
            prefix = []
            for component in id:
                prefix.append(component)
                if tuple(prefix) not in self.hier_sum:
                    self.hier_sum[tuple(prefix)] = 0
                if tuple(prefix) not in self.hier_cnt:
                    self.hier_cnt[tuple(prefix)] = 0
                self.hier_sum[tuple(prefix)] += self.elapsed[tuple(id)].microseconds
                self.hier_cnt[tuple(prefix)] += 1
                #print("key", tuple(prefix))

def record(path1: str, path2: str, kind: str, result: str, f):
    print(f'{path1},{path2},{kind},{result}', file=f)

def record_time(path1: str, path2: str, kind: str, elapsed: datetime.timedelta, f):
    return record(path1, path2, kind, str(elapsed.microseconds), f)

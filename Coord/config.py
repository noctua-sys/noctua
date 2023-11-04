import json
import functools

class Config:
    def __init__(self, num_sites, read_ops, write_ops, restriction_set):
        self.num_sites = num_sites
        self.read_ops = read_ops
        self.write_ops = write_ops
        self.op_set = read_ops + write_ops
        self.restriction_set = restriction_set

    @staticmethod
    def from_file(path):
        with open(path, 'r') as f:
            doc = json.loads(f.read())
            num_sites = int(doc['num_sites'])
            read_ops = doc['read_ops']
            write_ops = doc['write_ops']
            restriction_set = list()
            for [p, q] in doc['restriction_set']:
                restriction_set.append((p,q))
            return Config(num_sites, read_ops, write_ops, restriction_set)

    @functools.cache
    def conflicting_set(self, op: str) -> list[str]:
        cs = []
        for r in self.restriction_set:
            if r[0] == op: cs.append(r[1])
            elif r[1] == op: cs.append(r[0])
        return cs

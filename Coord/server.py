#!/usr/bin/env python3

import logging
from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer
from Coord.config import Config
import argparse

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Server:
    def __init__(self, config_file, sc):
        self.num_sites = 0
        self.table = dict()
        self.config = Config.from_file(config_file)
        self.sc = sc
        self.tick = 0

    def add(self, site, p):
        """Site try to start op.

        If there are conflicting operations, return False, and nothing is added.
        Otherwise, op is added, and an ID is returned."""

        assert isinstance(site, int)
        assert isinstance(p, str)

        self.tick += 1

        if self.sc:
            # Strong Consistency
            if len(self.table) > 0:
                return False
            else:
                self.table[self.tick] = (site,p)
                logging.info(f'+ {site} adds {p})')
                return self.tick

        else:
            # Relaxed Consistency
            cs = self.config.conflicting_set(p)

            # if any running op q conflicts with p
            for (_, q) in self.table.values():
                if q in cs:
                    return False

            self.table[self.tick] = (site,p)
            logging.info(f'+ {site} adds {p})')
            return self.tick

    def remove(self, id):
        """Site finished op, and the effects are already replicated."""
        try:
            (site,op) = self.table[id]
            logging.info(f'- {site} removes {op})')
            del self.table[id]
            return True
        except KeyError:
            return False

    def serve(self):
        logging.info(f"starting coordinator server")
        if self.sc: logging.info('... in strong consistency mode')
        logging.info(f"num_sites = {self.config.num_sites}")
        logging.info(f"op_set = {self.config.op_set}")
        logging.info(f"restriction_set = {self.config.restriction_set}")
        server = SimpleJSONRPCServer(('localhost', 4000))
        server.register_function(self.add, 'add')
        server.register_function(self.remove, 'remove')
        server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server')
    parser.add_argument('--config', type=str, help='path to config file', required=True)
    parser.add_argument('--sc', action='store_true', help='strong consistency')
    args = parser.parse_args()
    server = Server(args.config, args.sc)
    server.serve()

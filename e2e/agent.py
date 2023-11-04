# client emulator

import argparse
import json
import time

import requests
import random
import string
from Coord.client import coord

class Agent:
    def __init__(self, site, cfg, write_ratio, sc):
        self.site = site
        self.cfg = cfg
        self.tick = 0
        self.timing_table = dict()
        self.session = requests.Session()
        self.jwt_token = None
        self.write_ratio = write_ratio
        self.sc = sc
        self.read_ops = set(self.cfg['read_ops'])


    def gen_data(self, val_spec):
        """
        . randshortstr
        . randint MAX?  MAX is 10000 if absent
        . randdate  YYYY-MM-DD
        """

        # make contention more intense

        if val_spec.startswith('randshortstr'):
            return str(random.randint(1, 15))
        elif val_spec.startswith('randint'):
            parts = val_spec.split(' ')
            if len(parts) > 1:
                max_val = int(parts[1])
            else:
                max_val = 15
            return random.randint(1, max_val)
        elif val_spec.startswith('randdate'):
            return '2021-01-01'
        elif val_spec.startswith('pick'):
            return random.choice(val_spec.split(' ')[1:])
        elif val_spec.startswith('choice'):
            return random.choice(val_spec.split(' ')[1:])
        else:
            raise ValueError(f'unknown val_spec: {val_spec}')


    def do_request(self, url, method, data):

        # add jwt token if it is not none
        if self.jwt_token:
            headers = {'Authorization': f'JWT {self.jwt_token}'}
        else:
            headers =  {}

        if method == 'POST':
            r = self.session.post(url, data=data, headers=headers)
        elif method == 'GET':
            r = self.session.get(url, headers=headers)
        elif method == 'PUT':
            r = self.session.put(url, data=data, headers=headers)
        elif method == 'DELETE':
            r = self.session.delete(url, headers=headers)
        elif method == 'PATCH':
            r = self.session.patch(url, data=data, headers=headers)
        else:
            raise ValueError(f'unknown method: {method}')

        print(f'AGENT: {r}')

        return r


    def request(self, op_name):
        """
        Perform a request.
        """

        op = self.cfg['endpoints'][op_name]
        method = op['method']

        url = f'http://127.0.0.1:{4000 + self.site}/' + op['url']
        if '{}' in url:
            url = url.format(*[self.gen_data(val_spec) for val_spec in op['query_params']])

        data = dict()
        for k, v in op.get('data', dict()).items():
            data[k] = self.gen_data(v)

        # Time the request
        start = time.time()
        reqid = None
        if self.sc and op_name in self.read_ops:  # The django integration only coordinates write ops
            while reqid is None:
                time.sleep(0.001)
                reqid = coord.add(self.site, op_name)
        try:
            print(f'AGENT: request {op_name} at {url} with {data}')
            self.do_request(url, method, data)
        except Exception as e:
            print(e)
            pass
        if reqid is not None:
            coord.remove(reqid)
        end = time.time()
        elapsed_milliseconds = (end - start) * 1000

        self.tick += 1
        self.timing_table[(self.tick, op_name)] = elapsed_milliseconds

    def login(self):
        if 'login' in self.cfg:
            username = self.cfg['login']['username']
            password = self.cfg['login']['password']
            url = f'http://127.0.0.1:{4000 + self.site}/' + self.cfg['login']['url']
            # get jwt token
            self.jwt_token = self.session.post(url, data={'username': username, 'password': password}).json()['token']
            print('AGENT: login success')


    def run_for_secs(self, secs):
        """
        Randomly pick an operation and perform it.
        """

        self.login()

        self.secs = secs

        self.start_time = time.time()
        while time.time() - self.start_time <= secs:
            if random.random() < self.write_ratio:
                op_name = random.choice(self.cfg['write_ops'])
            else:
                op_name = random.choice(self.cfg['read_ops'])
            self.request(op_name)

    def save_stats(self, suffix):

        write_ops = set(self.cfg['write_ops'])
        read_ops = set(self.cfg['read_ops'])

        n_writes = 0
        n_txs = 0

        with open(f'{self.site}-{suffix}.csv', 'w') as f:
            for ((tick, op_name), elapsed_milliseconds) in self.timing_table.items():
                n_txs += 1
                if op_name in write_ops:
                    op_type = 'write'
                    n_writes += 1
                elif op_name in read_ops:
                    op_type = 'read'
                else:
                    raise ValueError(f'unknown op_name: {op_name}')
                f.write(f'{tick},{op_type},{op_name},{elapsed_milliseconds}\n')

        tps = n_txs / self.secs
        print('TPS =', tps)

        latency = sum(self.timing_table.values()) / n_txs
        print('average latency =', latency, 'ms')

        print('real write ratio =', n_writes / n_txs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Client')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--write-ratio', type=float, required=True)
    parser.add_argument('--secs', type=float, default=10)
    parser.add_argument('--site', type=int, required=True)
    parser.add_argument('--sc', action='store_true', default=False)
    args = parser.parse_args()

    agent = Agent(args.site, json.loads(open(args.config, 'r').read()), args.write_ratio, args.sc)
    agent.run_for_secs(args.secs)
    agent.save_stats(args.suffix)

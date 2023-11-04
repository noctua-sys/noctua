import re
import pandas as pd
from typing import Callable
from functools import *
from matplotlib import pyplot as plt
import numpy as np
import glob
import os
import os.path
import argparse


class SuffixedLog:
    def __init__(self, logdir: str, suffix: str = ''):
        self.logdir = logdir
        self.suffix = suffix
        self.load_trace()
        self.load_verify()
        self.load_analyze()
        self.effectful_path_ids = set(self.verify_df['path1'])

    def load_trace(self):
        """load trace"""
        self.num_explored_paths = self._find_number_in_trace(r'^Total number of explored paths.* ([0-9]+)$')
        self.num_effectful_paths = self._find_number_in_trace(r'^Total number of effectful paths.* ([0-9]+)$')
        self.time_find_static_info = self._find_number_in_trace(r'^Total time of Finding models and relations.* ([0-9\.]+)$')
        self.num_django_apps, self.num_models, self.num_fields = self._find_number_in_trace(r'^// (\d+) apps, (\d+) models \((\d+) fields\)')
        self.num_relations, self.num_oneone, self.num_manyone, self.num_manymany = self._find_number_in_trace(r'^// (\d+) relations \((\d+) oneone, (\d+) manyone, (\d+) manymany\)$')

    def _find_number_in_trace(self, pat):
        with open(self.logdir + '/trace' + self.suffix) as f:
            contents = f.read()
            res = []
            for match in re.findall(pat, contents, re.MULTILINE):
                if isinstance(match, tuple):
                    matches = match
                    for match in matches:
                        if '.' in match:
                            res.append(float(match))
                        else:
                            res.append(int(match))
                else:
                    if '.' in match:
                        res.append(float(match))
                    else:
                        res.append(int(match))
            return res

    def load_verify(self):
        """load verify.csv"""
        df = pd.read_csv(self.logdir + '/verify' + self.suffix + '.csv')
        df = df.sort_values(['path1', 'path2'])
        self.verify_df = df

    @property
    def num_checks(self):
        return len(self.verify_df)

    @property
    def num_restrictions(self):
        return len(self.verify_df[self.verify_df['restricted'] == True])

    @cached_property
    def total_verify_time(self):
        return self.verify_df['total_time'].sum()

    @cached_property
    def total_com_time(self):
        return self.verify_df['com_time'].sum()

    @cached_property
    def total_sem_time(self):
        return self.verify_df['sem_time'].sum()

    @cached_property
    def avg_com_time(self):
        return self.verify_df['com_time'].mean()

    @cached_property
    def avg_sem_time(self):
        return self.verify_df['sem_time'].mean()

    def load_analyze(self):
        """load analyze.csv"""
        df = pd.read_csv(self.logdir + '/analyze' + self.suffix + '.csv')
        df = df.sort_values('func')
        self.analyze_df = df

    @cached_property
    def total_explore_time(self):
        return self.analyze_df['time'].sum()

    def count_restrictions_involving(self, path):
        df = self.verify_df[self.verify_df['restricted']==True]
        self_conflict = df[(df['path1']==path) & (df['path2']==path)]
        first_conflict = df[(df['path1']==path)]
        second_conflict = df[(df['path2']==path)]
        num_restrictions = len(first_conflict) + len(second_conflict) - len(self_conflict)
        return num_restrictions

    def compute_core(self):
        df = self.verify_df[['path1', 'path2']].copy()
        df['num_restrictions'] = df.apply(lambda s: self.count_restrictions_involving(s['path1']), axis=1)
        df['restriction_ratio'] = df.apply(lambda s: s['num_restrictions'] / len(df), axis=1)
        df = df[['path1', 'num_restrictions', 'restriction_ratio']]
        df = df.rename(columns={'path1': 'path'})
        df.sort_values(by='num_restrictions', ascending=False)
        return df

    @cached_property
    def core(self):
        return self.compute_core()

    @lru_cache()
    def pair_is_true_on(self, field, path1, path2):
        df = self.verify_df
        df = df[df[field] == True]
        return len(df[(df['path1'] == path1) & (df['path2'] == path2)]) > 0 or len(df[(df['path2'] == path1) & (df['path1'] == path2)]) > 0

    def pair_is_restricted(self, path1, path2):
        return self.pair_is_true_on('restricted', path1, path2)

    def pair_is_com(self, path1, path2):
        return self.pair_is_true_on('com', path1, path2)

    def pair_is_sem(self, path1, path2):
        return self.pair_is_true_on('sem', path1, path2)

    def pair_is_indep(self, path1, path2):
        return self.pair_is_true_on('indep', path1, path2)

    def compute_conflict_table(self):
        paths = list(set(self.verify_df['path1']))  # must dedup
        n = len(paths)
        mat = np.empty((n,n), dtype=bool)
        for i in range(len(paths)):
            for j in range(len(paths)):
                mat[i,j] = self.pair_is_restricted(paths[i], paths[j])
        df = pd.DataFrame(mat, columns=paths, index=paths)
        return df

    @cached_property
    def conflict_table(self):
        return self.compute_conflict_table()

    def compute_comsem_table(self):
        paths = list(set(self.verify_df['path1']))  # must dedup
        n = len(paths)
        mat = np.empty((n,n), dtype="U20")   # numpy expects each cell is fixed-length
        for i in range(len(paths)):
            for j in range(len(paths)):
                com = '' if self.pair_is_com(paths[i], paths[j]) else '!com'
                sem = '' if self.pair_is_sem(paths[i], paths[j]) else '!sem'
                mat[i,j] = ' '.join([com, sem])
        df = pd.DataFrame(mat, columns=paths, index=paths)
        return df

    @cached_property
    def comsem_table(self):
        return self.compute_comsem_table()


def read_app_dir(dir):
    suffixes = [os.path.basename(f)[5:] for f in glob.glob(f'{dir}/trace*')]
    timeouts = [float(suffix[:-1]) for suffix in suffixes]
    n_runs = len(suffixes)
    logs = [SuffixedLog(dir, suffix) for suffix in suffixes]
    return logs, timeouts


def plot_app_timeouts(dir, app_name=None):
    logs, timeouts = read_app_dir(dir)
    df = pd.DataFrame({
        'explore_time': [log.total_explore_time for log in logs],
        'com_time': [log.total_com_time for log in logs],
        'sem_time': [log.total_sem_time for log in logs]
    }, index=timeouts)
    df = df.sort_index()
    title = 'Time breakdown'
    if app_name:
        title += ' for ' + app_name
    df.plot(kind='bar', stacked=True, legend=True, title=title, xlabel='Timeout (s)', ylabel='Time (s)')


def compare_app_timeouts(dir, app_name=None):
    logs, timeouts = read_app_dir(dir)
    log_plus_timeout = list(zip(logs, timeouts))
    log_plus_timeout.sort(key=lambda x: x[1])
    logs = [x[0] for x in log_plus_timeout]
    timeouts = [x[1] for x in log_plus_timeout]

    dfs = [log.verify_df[['path1', 'path2', 'com', 'sem', 'restricted']] for log in logs]
    for i in range(len(logs)):
        for j in range(i+1, len(logs)):
            dfi = dfs[i].set_index(['path1', 'path2'])
            dfj = dfs[j].set_index(['path1', 'path2'])
            dfi_only_idx = dfi.index.difference(dfj.index)
            dfj_only_idx = dfj.index.difference(dfi.index)
            both_idx = dfi.index.intersection(dfj.index)
            dfi_common = dfi.loc[both_idx]
            dfj_common = dfj.loc[both_idx]
            common_diff = dfi_common.compare(dfj_common)
            dfi_only = dfi.loc[dfi_only_idx]
            dfj_only = dfj.loc[dfj_only_idx]
            dfi_restricted = dfi[dfi['restricted'] == True].index.intersection(dfj.index)
            dfj_restricted = dfj[dfj['restricted'] == True].index.intersection(dfi.index)
            has_diff = len(dfi_only) > 0 or len(dfj_only) > 0 or len(common_diff) > 0
            if has_diff:
                print(f'* %sCompare {timeouts[i]}s and {timeouts[j]}s' % ("" if not app_name else app_name + " "))
                print(f'For the common rows:')
                print(f'- Only restricted in {timeouts[i]}: ', dfi_restricted.difference(dfj_restricted).to_list())
                print(f'- Only restricted in {timeouts[j]}: ', dfj_restricted.difference(dfi_restricted).to_list())
            if len(dfi_only) > 0:
                print(f'** only in {timeouts[i]}s')
                print(dfi_only.to_markdown(index=True))
            if len(dfj_only) > 0:
                print(f'** only in {timeouts[j]}s')
                print(dfj_only.to_markdown(index=True))
            if len(common_diff) > 0:
                print(f'** common parts of {timeouts[i]}s and {timeouts[j]}s differences')
                print(common_diff.to_markdown(index=True))


def app_log_to_stats(log, index):
    return pd.DataFrame({
        'num_models': log.num_models,
        'num_relations': log.num_relations,
        'num_explored_paths': log.num_explored_paths,
        'num_effectful_paths': log.num_effectful_paths,
        'time_find_static_info': log.time_find_static_info,
        'time_explore': log.total_explore_time,
        'time_com': log.total_com_time,
        'time_sem': log.total_sem_time,
        'num_checks': log.num_checks,
        'restrictions': log.num_restrictions
    }, index=[index])


def table_for_app_statistics(dir):
    dfs = []
    for app in [os.path.basename(d) for d in glob.glob(f'{dir}/*')]:
        app_dir = f'{dir}/{app}'
        logs, timeouts = read_app_dir(app_dir)
        for log,timeout in zip(logs,timeouts):
            dfs.append(app_log_to_stats(log, f'{app}-{timeout}s'))
    df = pd.concat(dfs)
    return df


def read_all_logs(dir):
    ret = []
    for app in [os.path.basename(d) for d in glob.glob(f'{dir}/*')]:
        app_dir = f'{dir}/{app}'
        logs, timeouts = read_app_dir(app_dir)
        for log,timeout in zip(logs,timeouts):
            ret.append((log, app, timeout))
    return ret


def plot_all_timeouts(dir):
    dfs = []
    for (log, app, timeout) in read_all_logs(dir):
        df = pd.DataFrame({
            'explore_time': [log.total_explore_time],
            'com_time': [log.total_com_time],
            'sem_time': [log.total_sem_time],
        }, index=[f'{app}-{timeout}'])
        dfs.append(df)
    df = pd.concat(dfs)
    df.plot(kind='bar', stacked=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='命令行程序')

    parser.add_argument('command', metavar='COMMAND', type=str, nargs=1, help='{gen-apps-table | compare-app-timeouts | plot-app-timeouts}')
    parser.add_argument('-d', '--dir', metavar='DIRECTORY', type=str, default=os.getcwd(),
                        help='Specify the directory, default=CWD')
    parser.add_argument('-o', '--output', metavar='OUTPUT', type=str, default=None,
                        help='Specify the path of the output file, default=None')

    args = parser.parse_args()

    command = args.command[0]

    if command == 'help':
        parser.print_help()
    elif command == 'gen-apps-table' or command == 'apps-table':
        df = table_for_app_statistics(args.dir)
        if not args.output:
            print(df.to_markdown())
        elif args.output[-3:] == 'xls' or args.output[-4:] == 'xlsx':
            df.to_excel(args.output)
        elif args.output[-3:] == 'csv':
            df.to_csv(args.output, index=False)
        else:
            print('Unrecognized format: ' + args.output)
    elif command == 'compare-app-timeouts':
        compare_app_timeouts(args.dir)
    elif command == 'plot-all-timeouts':
        plot_all_timeouts(args.dir)
        if args.output:
            plt.savefig(args.output)
        else:
            plt.show()
    elif command == 'plot-app-timeouts':
        plot_app_timeouts(args.dir)
        if args.output:
            plt.savefig(args.output)
        else:
            plt.show()
    elif command == 'uc':
        logs, timeouts = read_app_dir(args.dir)
        for i in range(len(timeouts)):
            print(f'* for timeout {timeouts[i]}s')
            df = logs[i].compute_core()
            print(df.to_markdown(index=False))
    elif command == 'conflict-table':
        logs, timeouts = read_app_dir(args.dir)
        for i in range(len(timeouts)):
            df = logs[i].compute_conflict_table()
            df.to_csv(f'conflict-table{timeouts[i]}.csv', index=True)
    elif command == 'comsem-table':
        logs, timeouts = read_app_dir(args.dir)
        for i in range(len(timeouts)):
            df = logs[i].compute_comsem_table()
            df.to_csv(f'comsem-table{timeouts[i]}.csv', index=True)
    else:
        print("Unknown commands")


# debug-only
if '__PYTHON_EL_get_completions' in dir():
    import os
    home = os.getenv('HOME')
    log = SuffixedLog(home + '/src/soir/apps/TIMELOGS/20230316-19:21:53/PostGraduation', '2s')

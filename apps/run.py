#!/usr/bin/env python3

"""Consistency Analysis Runner.

This file runs the consistency analysis on all supported applications
(see APPS) with a given set of timeouts (see TIMEOUTS) in parallel.

"""

from dataclasses import dataclass
import datetime
import concurrent.futures
import subprocess
import logging
import pathlib
import os

logging.basicConfig(
    level=logging.INFO
)


@dataclass
class App:
    """The base class of all applications to be evaluated."""

    name: str
    path: str

    def pre(self, job: "Job"):
        """Override this method to customize job.pre()."""
        pass

    def post(self, job: "Job"):
        """Override this method to customize job.post()."""
        pass


@dataclass
class Job:
    """A Job object represents a concrete measurement job to be run.

    The Job class should not be inherited.

    """

    app: App
    timeout: float
    timestamp: str

    def pre(self):
        self.app.pre(self)

    def post(self):
        self.app.post(self)

    @property
    def name(self):
        return f"{self.app.name}-{self.timeout}"

    def __repr__(self):
        return self.name

    def async_run_on(self, executor):
        """Start a job, and returns a Future object."""
        return executor.submit(self.blocking_run)

    def blocking_run(self):
        """On the current thread, start the job and wait for its completion."""
        suffix = f"{self.timeout}s"
        timeout = int(self.timeout)
        dirparam = f"logs/{self.timestamp}/"
        outdir = f"{self.app.path}/{dirparam}"
        self.outdir = outdir

        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        logfile = open(f'{outdir}/trace{suffix}', 'w')
        logging.info(f"Starting job {self.name} at {outdir}")
        # [KeyboardInterrupt] won't be raised when ^C is signaled.  SIGINT will be
        # sent to the child processes, and handled there.
        p = subprocess.Popen(
            ["python", "manage.py", "consistency",
             f"--suffix={suffix}",
             f"--timeout={timeout}",
             f"--dir={dirparam}"],
            cwd=self.app.path,
            stdin=subprocess.DEVNULL,
            stdout=logfile,
            stderr=subprocess.STDOUT,
        )
        status = p.wait()
        logfile.close()
        if status != 0:
            logging.warning(f"Job {self.name} failed with exit code {status}")
        else:
            logging.info(f"Job {self.name} finished successfully")
        return status


APPS = [
    App("zhihu", "zhihu/backend.orig"),
    App("PostGraduation", "PostGraduation.orig"),
    #App("django-todo", "django-todo"),
    #App("ownphotos", "ownphotos"),
    App('smallbank', 'benchmark_smallbank'),
    App('courseware', 'benchmark_courseware'),
    #App('seats', 'benchmark_seats')
]
# TIMEOUTS = [ 1,2,3,4,5,6,7,8,9,10 ]
# TIMEOUTS = [ 0.1 ]
TIMEOUTS = [2]
MAX_WORKERS = 4

def run_all_jobs_parallel(timestamp):
    """Create a sized threadpool and run all jobs in this pool."""
    logging.info("Start running all jobs")
    logging.info(f"Applications: {APPS}")
    logging.info(f"Timeouts: {TIMEOUTS}")
    logging.info(f"Timestamp: {timestamp}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for app in APPS:
            for timeout in TIMEOUTS:
                job = Job(app, timeout, timestamp)
                job.async_run_on(executor)

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')
    run_all_jobs_parallel(timestamp)

    # Link from LOGS/{app}/{time} to TIMELOGS/{time}/{app}
    pathlib.Path(f'TIMELOGS/{timestamp}').mkdir(parents=True, exist_ok=True)
    for app in APPS:
        os.symlink(f'../../LOGS/{app.name}/{timestamp}', f'TIMELOGS/{timestamp}/{app.name}')

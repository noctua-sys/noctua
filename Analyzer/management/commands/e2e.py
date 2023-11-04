
import traceback
import re
import sys
import pathlib, importlib, importlib.util, inspect
from typing import Iterable, Tuple
from Analyzer.utils import simple_field_to_type
from Verifier.timeout import WorkerProcessError
from django.db import models
from django.core.management.base import BaseCommand, CommandError
from django.core.exceptions import ViewDoesNotExist
from django.contrib.admindocs.views import simplify_regex
from django.urls import URLPattern, URLResolver
from django.conf import settings
from django.http import HttpRequest
from django.middleware import csrf
from Analyzer.symbolic import *
from Analyzer.analyzer import PathFinder
from Analyzer.translator import IRprinter
from Verifier.checker import *
from Analyzer.management.commands import result_summary


import time
import pandas as pd

from e2e import basic
from django.core.management.commands.runserver import Command as RunserverCommand

class Command(BaseCommand):
    help = 'runserver with e2e parameters filled'

    def __init__(self):
        super().__init__()

    def add_arguments(self, parser):
        parser.add_argument(
            '--site', '-s', dest='site', required=True,
            help='which site am I?'
        )

    def handle(self, *args, **options):
        basic._current_site = int(options['site'])
        print('e2e: I am site %d' % basic.current_site())

        command = RunserverCommand()
        command.handle(
            use_ipv6=False,
            addrport=str(4000+basic.current_site()),
            use_reloader=True,
            use_threading=True,
        )



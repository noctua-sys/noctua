"""
Credit: This module is an adaption of show_urls from django-extensions [1].

[1] https://github.com/django-extensions/django-extensions
"""

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


class RegexURLPattern: pass
class RegexURLResolver: pass
class LocaleRegexURLResolver: pass


def describe_pattern(p):
    return str(p.pattern)


class FakeSession:
    def flush(self):
        print('session flushed')

    def get(self, attr):
        return None


class FakeParams:
    def __init__(self, underlying_dict=None):
        self.underlying_dict = underlying_dict

    def get(self, key, default=None):
        if key in self.underlying_dict:
            return key

        import Analyzer.notify as notify
        has = SymBool(notify.obtain_free_expr('arg_POST_has_' + str(key), Type.BOOL()))
        if has:
            return self[key]
        else:
            return default

    def __getitem__(self, key):
        if key in self.underlying_dict:
            return key
        else:
            # FIXME: APP-SPECIFIC
            if key == 'title' or key == 'body' or key == 'isCompleted' or key == 'name':
                ty = Type.STR()
            elif key == 'amount':
                ty = Type.REAL()
            else:
                ty = Type.INT()
            return self._getitem_with_type(key, ty)

    def _getitem_with_type(self, key, typ: Type):
        """This is similar to __getitem__, but takes an additional parameter for argument type."""
        assert isinstance(typ, Type)
        import Analyzer.notify as notify
        return Sym(notify.obtain_free_expr('arg_POST_' + str(key), typ), typ.pytype())

    def _getitem_with_pytype(self, key, typ: type):
        """This is similar to _getitem_with_type, but @typ is a Python type object."""
        return self._getitem_with_type(key, Type.from_pytype(typ))

    def __len__(self):
        # FIXME: this is not correct generally.
        return len(self.underlying_dict)


# FakeRequest
#
# request.POST['abc'] -> Symbolic('abc123')
# new argument: abc
class FakeRequest(HttpRequest):
    """A Symbolic request object."""

    def __init__(self, method):
        super().__init__()

        csrf_token = csrf._get_new_csrf_token()
        self.method = method
        self.META = {
            'SERVER_NAME': 'localhost',
            'SERVER_PORT': '12345',

            # CSRF
            'CSRF_COOKIE': csrf_token,
            settings.CSRF_HEADER_NAME: csrf_token
        }
        self.POST = FakeParams({
            'csrfmiddlewaretoken': csrf_token
        })
        self.session = FakeSession()

    @property
    def user(self):
        """See also patch_drf.Request.user."""
        if hasattr(self, '_user'):
            return self._user
        from Analyzer.patch_django import FakeUser
        self._user = FakeUser()
        return self._user

def extract_kwargs(regex):
    pattern = simplify_regex(regex)
    kwargs = []
    while True:
        start = pattern.find('<')
        if start == -1:
            return kwargs
        end = pattern.find('>')
        kwargs.append(pattern[start+1:end])
        pattern = pattern[end+1:]


def get_kwargs_types(kwargs, url):
    d = {}
    for name in kwargs:
        if ':' in name:
            [type, name] = name.split(':')
            if type == 'int':
                d[name] = Type.INT()
                continue
            elif type == 'str':
                d[name] = Type.STR()
                continue
            else:
                print('exporturls: unknown type: {}'.format(type))

        # Has no :, or unknown type signature
        # FIXME: APP-SPECIFIC
        if 'photos' in url and name == 'pk':
            d[name] = Type.STR()
        elif name in ['pk', 'answer', 'question', 'id']:
            d[name] = Type.INT()
        else:
            d[name] = Type.STR()
    return d


def make_symbolic_kwargs(kwargs_dict):
    symbolic_classes = {
        'int': SymInt,
        'str': SymStr
    }
    ret = {}
    for name, ty in kwargs_dict.items():
        symval = symbolic_classes[str(ty)](Free('arg_' + name, ty.smt(None), ty))
        ret[name] = symval
    return ret


class Command(BaseCommand):
    help = 'Collect effects and do the verification.'

    def __init__(self):
        super().__init__()
        self.models = {}
        self.relations = []
        self.app_cnt = 0
        self.model_cnt = 0
        self.field_cnt = 0
        self.relation_cnt = 0
        self.relation_oneone_cnt = 0
        self.relation_manyone_cnt = 0
        self.relation_manymany_cnt = 0

        # Relation fields can refer to models that are not yet declared.
        # We delay the processing of relation fields until all models are discovered.
        self.delayed_relation_field_closures = []

    def add_arguments(self, parser):
        parser.add_argument(
            '--urlconf', '-c', dest='urlconf', default='ROOT_URLCONF',
            help='Set the settings URL conf variable to use'
        )
        parser.add_argument(
            '--suffix', '-s', dest='suffix', default='',
            help='Suffix to output files'
        )
        parser.add_argument(
            '--timeout', '-T', dest='timeout', default='1',
            help='Timeout of the verification of a pair of ops (units = second)'
        )
        parser.add_argument(
            '--dir', '-d', dest='dir', default='.',
            help='Output directory of output files'
        )
        parser.add_argument(
            '--analyze-regexp', '-A', dest='analyze_regexp', default='.*',
            help='IDs of operations to analyze must match this regexp'
        )
        parser.add_argument(
            '--verify-regexp', '-V', dest='verify_regexp', default='.*',
            help='IDs of operations to verify must match this regexp'
        )
        parser.add_argument(
            '--save-regexp', '-S', dest='save_regexp', default=None,
            help='If IDs of operations match this regexp, an SMTLib2 file is saved'
        )
        parser.add_argument(
            '--save-filename-format', '-F', dest='save_filename_format', default='{kind}-{id1}-{id2}-{suffix}.smt2'
        )
        parser.add_argument(
            '--independence', '-i', dest='check_independence', default=False,
            help='Also check the independence rule'
        )

    def handle(self, *args, **options):
        self.suffix = options['suffix']
        self.timeout = float(options['timeout'])
        self.dir = options['dir']
        self.analyze_regexp = re.compile(options['analyze_regexp'])
        self.verify_regexp = options['verify_regexp']
        self.save_regexp = options['save_regexp']
        self.save_filename_format = options['save_filename_format']
        self.check_independence = options['check_independence']

        # Load urlconf
        urlconf = options['urlconf']
        if not hasattr(settings, urlconf):
            raise CommandError('Settings module {} does not have the attribute {}.'.format(settings, urlconf))
        try:
            urlconf = __import__(getattr(settings, urlconf), {}, {}, [''])
        except Exception as e:
            if options['traceback']:
                traceback.print_exc()
            raise CommandError('Error occurred while trying to load %s: %s' % (getattr(settings, urlconf), str(e)))

        self.time_counter = result_summary.TimeCounter()

        # Collect static information
        with self.time_counter.time("Find models and relations"):
            self.find_models_and_relations()
            self.print_result()

        # Collect dynamic information
        with self.time_counter.time("Collect all paths"):
            summary = self.collect_paths(urlconf)
            with open(f'{self.dir}/export{self.suffix}.org', 'w') as f:
                summary.pretty_print(f)

        # Do the verification.
        print('* Verification started, timeout =', self.timeout)
        with self.time_counter.time("Verify"):
            summary.verify(list(self.models.values()), self.relations, self.suffix, self.timeout)

        # Report results
        print("Total number of explored paths =", summary.total_number_of_explored_paths)
        print("Total number of effectful paths =", summary.total_number_of_effectful_paths)
        print("Total time of Finding models and relations", self.time_counter.elapsed["Find models and relations"].microseconds)
        print("Total time of Collecting all paths", self.time_counter.elapsed["Collect all paths"].microseconds)

    def print_result(self):
        def print_field(f: Field):
            print(f'   {f.name}: {f.ty} [{",".join(map(lambda x: str(x), f.attrs))}];')
        def print_model(m: Model):
            assert isinstance(m, Model)
            print('model %s {' % m.name)
            for f in m.fields.values():
                print_field(f)
            print('}')
        def print_relation(r: Relation):
            assert isinstance(r, Relation)
            print('relation {} {} {} {};'.format(r.name, r.kind, r.from_, r.to_))
        for m in self.models.values():
            print_model(m)
        for r in self.relations:
            print_relation(r)
        print('// {} apps, {} models ({} fields)'.format(self.app_cnt, self.model_cnt, self.field_cnt))
        print('// {} relations ({} oneone, {} manyone, {} manymany)'.format(self.relation_cnt, self.relation_oneone_cnt, self.relation_manyone_cnt, self.relation_manymany_cnt))

    def find_models_and_relations(self):
        project_dir = pathlib.Path(settings.BASE_DIR)
        assert project_dir.is_dir() and project_dir.exists()
        self.stderr.write(self.style.SUCCESS('* Analyzing project under {}'.format(project_dir)))

        for app in settings.INSTALLED_APPS:
            if app in ['Analyzer', 'Verifier'] or app.startswith('django'):
                continue
            try:
                spec = importlib.util.find_spec(app)
                pathlib.Path(spec.origin).relative_to(project_dir)
                module = importlib.import_module(app + '.models')
            except:
                self.stderr.write(self.style.ERROR('Failed to load models from app {}'.format(app)))
                # traceback.print_exc()
                continue
            self.stderr.write(self.style.SUCCESS('  * Analyzing app {}'.format(app)))
            self.app_cnt += 1
            for attr in dir(module):
                item = getattr(module, attr)
                if inspect.isclass(item) and issubclass(item, models.Model):
                    label = item._meta.label
                    self.stderr.write(self.style.SUCCESS('    * Analyzing model {}'.format(label)))
                    self.process_model(item)

        for f in self.delayed_relation_field_closures:
            f()

    def process_model(self, model_class):
        fields = []
        for field in model_class._meta.get_fields():
            if field.is_relation:
                def mk_f(field):  # mk_f is necessary to keep the correct field object.
                    def f():
                        try:
                            self.process_relation(field, from_model=model_class)
                        except KeyError:
                            self.stderr.write(self.style.ERROR('Failed to process relation field {}'.format(field)))
                    return f
                self.delayed_relation_field_closures.append(mk_f(field))
            else:
                fields.append(self.process_field(field))
        self.model_cnt += 1
        model = Model(model_class._meta.label, fields)
        self.models[model.name] = model

    def process_field(self, field):
        f = Field(field.name, simple_field_to_type(field), [])

        if f.ty is None:
            raise RuntimeError('does not support field type: {}'.format(field.__class__.__name__))

        # attributes.
        if field.primary_key:
            f.attrs.append(Attr.PRIMARY)
        if field.unique:
            f.attrs.append(Attr.UNIQUE)
        if field.null:
            f.attrs.append(Attr.OPTIONAL)

        self.field_cnt += 1

        return f

    def process_relation(self, relation, from_model):
        if isinstance(relation, models.OneToOneField):
            kind = RelationKind.ONE_ONE
            from_model, to_model = from_model._meta.label, relation.related_model._meta.label
            self.relation_oneone_cnt += 1
        elif isinstance(relation, models.ForeignKey):
            kind = RelationKind.MANY_ONE
            # If there is a foreign key K going from A to B,
            # then one B object can be associated with many A objects.
            # Therefore, 'Many' for A, 'One' for B.
            from_model, to_model = from_model._meta.label, relation.related_model._meta.label
            self.relation_manyone_cnt += 1
        elif isinstance(relation, models.ManyToManyField):
            kind = RelationKind.MANY_MANY
            from_model, to_model = from_model._meta.label, relation.related_model._meta.label
            self.relation_manymany_cnt += 1
        elif isinstance(relation, models.ManyToManyRel):
            kind = RelationKind.MANY_MANY
            from_model, to_model = from_model._meta.label, relation.related_model._meta.label
            self.relation_manymany_cnt += 1
        elif isinstance(relation, models.ManyToOneRel):
            # reverse lookup key
            return
        else:
            raise RuntimeError('relation kind not supported: {} is {}'.format(relation, relation.__class__.__name__))
        self.relation_cnt += 1
        r = Relation( '{}__{}__{}'.format(from_model, to_model, relation.name), kind, self.models[from_model], self.models[to_model])
        self.relations.append(r)

    def collect_paths(self, urlconf) -> "Summary":
        """
        Symbolically execute the view functions and turn them into a static summary.
        """
        views = self.extract_views_from_urlpatterns(urlconf.urlpatterns)
        summary = Summary(
            output_dir=self.dir,
            suffix=self.suffix,
            verify_regexp=self.verify_regexp,
            save_regexp=self.save_regexp,
            save_filename_format=self.save_filename_format,
            check_independence=self.check_independence,
        )

        df = pd.DataFrame(columns=['func', 'explored', 'effectful', 'time'])

        for (func, regex, url_name) in views:
            simplified = simplify_regex(regex)
            if simplified.startswith('/admin'):
                continue
            elif simplified.startswith('/api-auth'):
                continue
            elif 'register' in simplified:
                continue

            # Skip .<format> variant. It only affects the final output, not side effects.
            if '<format>' in simplify_regex(regex):
                continue
            print('')
            print(simplify_regex(regex), func, regex, url_name)

            methods = ['POST', 'DELETE']
            for method in methods:
                print('for %s...' % method, end='')

                func_id = make_func_id(simplify_regex(regex), method)
                if not self.analyze_regexp.match(method + '__' + func_id):
                    print('(ignored)')
                    continue

                # Symbolic arguments eventually added as arguments in
                # <PathFinder.before_invocation()>.
                kwargs = extract_kwargs(regex)
                kwargs_types = get_kwargs_types(kwargs, func_id)
                kwargs = make_symbolic_kwargs(kwargs_types)
                print(kwargs)

                request = FakeRequest(method)
                path_finder = PathFinder(func, extra_args=(request,), extra_kwargs=kwargs)
                start = time.perf_counter()
                path_finder.explore()
                explore_time = time.perf_counter() - start
                df.loc[len(df)] = [func_id, path_finder.number_of_explored_paths, len(path_finder.paths), explore_time]

                # Collect results
                summary.record_number_of_explored_paths(simplify_regex(regex),
                                                        method,
                                                        path_finder.number_of_explored_paths)
                if len(path_finder.paths) > 0:
                    summary.add_paths(simplify_regex(regex), method, path_finder.paths)

        df.to_csv(f'{self.dir}/analyze{self.suffix}.csv', index=False)
        return summary


    def extract_views_from_urlpatterns(self, urlpatterns, base='', namespace=None):
        """
        Return a list of views from a list of urlpatterns.

        Each object in the returned list is a three-tuple: (view_func, regex, name)
        """
        views = []
        for p in urlpatterns:
            if isinstance(p, (URLPattern, RegexURLPattern)):
                try:
                    if not p.name:
                        name = p.name
                    elif namespace:
                        name = '{0}:{1}'.format(namespace, p.name)
                    else:
                        name = p.name
                    pattern = describe_pattern(p)
                    views.append((p.callback, base + pattern, name))
                except ViewDoesNotExist:
                    continue
            elif isinstance(p, (URLResolver, RegexURLResolver)):
                try:
                    patterns = p.url_patterns
                except ImportError:
                    continue
                if namespace and p.namespace:
                    _namespace = '{0}:{1}'.format(namespace, p.namespace)
                else:
                    _namespace = (p.namespace or namespace)
                pattern = describe_pattern(p)
                if isinstance(p, LocaleRegexURLResolver):
                    for language in self.LANGUAGES:
                        with translation.override(language[0]):
                            views.extend(self.extract_views_from_urlpatterns(patterns, base + pattern, namespace=_namespace))
                else:
                    views.extend(self.extract_views_from_urlpatterns(patterns, base + pattern, namespace=_namespace))
            elif hasattr(p, '_get_callback'):
                try:
                    views.append((p._get_callback(), base + describe_pattern(p), p.name))
                except ViewDoesNotExist:
                    continue
            elif hasattr(p, 'url_patterns') or hasattr(p, '_get_url_patterns'):
                try:
                    patterns = p.url_patterns
                except ImportError:
                    continue
                views.extend(self.extract_views_from_urlpatterns(patterns, base + describe_pattern(p), namespace=namespace))
            else:
                raise TypeError("%s does not appear to be a urlpattern object" % p)
        return views


def make_func_id(id: str, method: str):
    return f'{method}__{id}'


class Summary:
    """Summary of the results of the code analysis."""

    def __init__(self,
                 output_dir: str,
                 suffix: str,
                 verify_regexp: str,
                 save_regexp: Optional[str],
                 save_filename_format: str,
                 check_independence: bool
                 ):
        # For each (id, HTTP method), there is a set of Paths.
        self.number_of_explored_paths: dict[Tuple[str, str], int] = {}
        self.paths: dict[Tuple[str, str], set[Path]] = {}
        self.total_number_of_explored_paths = 0
        self.total_number_of_effectful_paths = 0
        self.check_independence = check_independence

        self.output_dir = output_dir
        self.suffix = suffix
        self.verify_regexp = re.compile(verify_regexp)
        if save_regexp is None:
            self.save_regexp = None
        else:
            self.save_regexp = re.compile(save_regexp)
        self.save_filename_format = save_filename_format
        self.name_counter: dict[str, int] = dict()  # used to uniquify path names

    def uniquify(self, name):
        if name not in self.name_counter:
            self.name_counter[name] = 0
        self.name_counter[name] += 1
        return name + str(self.name_counter[name])

    def record_number_of_explored_paths(self, func_name, method, number):
        self.number_of_explored_paths[(func_name, method)] = number
        self.total_number_of_explored_paths += number

    def add_paths(self, id, method, paths: Iterable[Path]):
        self.paths.setdefault((id, method), set())
        for path in paths:
            self.paths[(id, method)].add(path)
            self.total_number_of_effectful_paths += 1

    def pretty_print(self, f=sys.stdout):
        print('#+startup: overview indent', file=f)
        for (id, method), paths in self.paths.items():
            print(f'* {method} {id} has {len(paths)} effectful paths', file=f)
            for i, path in enumerate(paths):
                print(f'** Path {i+1}', file=f)
                self.pp_args(path.free_vars, path.unique_ids, f)
                self.pp_cond(path.path_conds, f)
                self.pp_effs(path.effects, f)
                # self.pp_ir(path, f)

    def pp_args(self, args, unique_ids, f):
        print(f'*** Number of args = {len(args)}', file=f)
        for (arg, type) in args.items():
            if arg in unique_ids.keys():
                print('- ', arg, type, 'UNIQUE', file=f)
            else:
                print('- ', arg, type, file=f)

    def pp_cond(self, cond, f):
        print(f'*** The path condition is a conjuction of {len(cond)} clauses', file=f)
        for clause in list(cond):
            print('- ', clause, file=f)

    def pp_effs(self, effects, f):
        print(f'*** Number of {len(effects)} effects', file=f)
        for eff in effects:
            print('- ', eff, file=f)

    def pp_ir(self, path, f):
        path_IRprinter = IRprinter(path, f)
        path_IRprinter.printer()

    def make_id_readable_and_unique(self, id, method):
        maybe_dup = make_func_id(id, method)
        return self.uniquify(maybe_dup)

    def path_to_op(self, name: str, path: Path):
        class PathOp(Op):
            def __init__(self):
                super().__init__(
                    name=name,
                    argspecs=list(path.free_vars.items()),
                    body=None,
                    unique_ids=path.unique_ids,
                )

            def condition(self, context: 'Context', sys: 'SystemState', args) -> z3.BoolRef:
                import z3
                # All path conditions are evaluated against the initial condition.
                res = []
                for expr in path.path_conds:
                    res.append(expr.checked_eval(context, sys))
                return z3.And(res)

            def effect(self, context: 'Context', sys: 'SystemState', args, prefix=''):
                current_sys_state = sys
                for cmd in path.effects:
                    current_sys_state = cmd.apply(context, current_sys_state)[0]
                return current_sys_state
        return PathOp

    def verify(self, models, relations, suffix, timeout: float):
        # List all paths.
        all_paths = []
        for ((id, method), paths) in self.paths.items():
            for path in paths:
                all_paths.append((self.make_id_readable_and_unique(id, method), path))

        # List all pairs, including (x,x) where x=x.
        if self.check_independence:
            columns = ['path1', 'path2', 'com', 'sem', 'indep', 'restricted', 'com_time', 'sem_time', 'indep_time', 'total_time']
        else:
            columns = ['path1', 'path2', 'com', 'sem', 'restricted', 'com_time', 'sem_time', 'total_time']
        df = pd.DataFrame(columns=columns)
        num_paths = len(all_paths)
        total = (num_paths + 1) * num_paths / 2
        cur = 0
        for i in range(num_paths):
            for j in range(i, num_paths):
                cur += 1
                try:
                    id1, path1 = all_paths[i]
                    id2, path2 = all_paths[j]

                    # Skip checks that are excluded by verify_regexp.
                    if not self.verify_regexp.match(id1) and not self.verify_regexp.match(id2):
                        continue

                    # Set states for _save_file_hook.
                    self.cur_id1 = id1
                    self.cur_id2 = id2
                    checker = PairChecker(models, relations, timeout, self._save_file_hook)

                    op1 = self.path_to_op(id1, path1)
                    op2 = self.path_to_op(id2, path2)
                    P = op1()
                    Q = op2()
                    print(f'check pair ({cur}/{total} {cur/total*100:g}%): {P.name} {Q.name}')
                    sys.stdout.flush()

                    # com
                    start = time.perf_counter()
                    try:
                        com_res = checker.check_commutativity(P, Q)
                        print('   com ' + str(com_res))
                    except TimeoutError:
                        print(' ! com timeout'); sys.stdout.flush()
                        com_res = z3.unknown
                    com_time = time.perf_counter() - start

                    # sem
                    start = time.perf_counter()
                    try:
                        sem_res = checker.check_precondition(P, Q)
                        print('   sem ' + str(sem_res))
                    except TimeoutError:
                        print(' ! sem timeout'); sys.stdout.flush()
                        sem_res = z3.unknown
                    sem_time = time.perf_counter() - start

                    # indep
                    if self.check_independence:
                        start = time.perf_counter()
                        try:
                            indep_res = checker.check_independent(P, Q)
                            print('   indep ' + str(indep_res))
                        except TimeoutError:
                            print(' ! indep timeout'); sys.stdout.flush()
                            indep_res = z3.unknown
                        indep_time = time.perf_counter() - start

                    # record result
                    should_restrict = sem_res != z3.unsat or com_res != z3.unsat
                    if self.check_independence:
                        df.loc[len(df)] = [id1, id2,
                                           describe(com_res), describe(sem_res), describe(indep_res),
                                           should_restrict,
                                           com_time, sem_time, indep_time, com_time + sem_time + indep_time]
                    else:
                        df.loc[len(df)] = [id1, id2,
                                           describe(com_res), describe(sem_res), should_restrict,
                                           com_time, sem_time, com_time + sem_time]

                except WorkerProcessError as e:
                    print("  Cannot verify pair! due to exception: ")
                    traceback.print_exc()
                    sys.stdout.flush()
                except Exception as e:
                    print("  Cannot verify pair! due to exception: " + repr(e))
                    traceback.print_exc()
                    sys.stdout.flush()

        df.to_csv(f'{self.output_dir}/verify{suffix}.csv', index=False)


    def _save_file_hook(self, kind, solver):
        if not self.save_regexp or not self.save_regexp.match(self.cur_id1) or not self.save_regexp.match(self.cur_id2):
            return
        if '{suffix}' not in self.save_filename_format:
            self.save_filename_format += '{suffix}'
        filename = self.save_filename_format.format(id1=self.cur_id1, id2=self.cur_id2, kind=kind, suffix=self.suffix)
        filename = filename.replace('/', '!')
        path = f'{self.output_dir}/{filename}'
        with open(path, 'w') as f:
            f.write(solver.sexpr())
            f.write("\n(check-sat)\n(get-model)\n")

def describe(res):
    if res == z3.unsat:
        return 'proved'
    elif res == z3.sat:
        return 'falsifiable'
    else:
        return 'unknown'

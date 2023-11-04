import ast
import importlib
from rest_framework.viewsets import *

Comparision = ['>', '>=', '==', '<', '<=','in']

from Analyzer.path import _make_immutable


#@lru_cache(maxsize=None)
def parse_module(module_qname: str) -> ast.Module:
    pass

def translate_queryset(vs):
    if hasattr(vs, 'get_queryset'):
        return translate_queryset_func(vs)
    else:
        return translate_queryset_inline(vs)


def translate_queryset_inline(vs):
    '''
    queryset = Model.objects.qs
    '''
    module_qname = vs.__module__
    vs_qname = vs.__name__
    


def translate_queryset_func(vs):
    '''
    Overrides get_queryset.
    '''
    

def translate_create(vs):
    '''
    
    '''


class IRprinter:
    def __init__(self, path, file):
        self.path = path
        self.file = file

    def get_expressions(self, expr):
        print('IRDEBUG ', expr)
        if type(expr) == str:
            return expr
        if type(expr) == tuple:
            return str(expr)
        if expr[0] == 'deref':
            return self.get_deref(expr)
        if expr[0] == 'all':
            return self.get_all(expr)
        if expr[0] == 'exists':
            return self.get_exists(expr)
        if expr[0] == 'first':
            return self.get_first(expr)
        if expr[0] == 'orderby':
            return self.get_orderby(expr)
        if expr[0] == 'filter':
            return self.get_filter(expr)
        if expr[0] == 'limit':
            return self.get_limit(expr)
        if expr[0] == 'singleton':
            return self.get_singleton(expr)
        if expr[0] == 'project':
            return self.get_project(expr)
        if expr[0] == 'get':
            return self.get_get(expr)
        if expr[0] == 'link':
            return self.get_link(expr)
        if '.' in expr[0]:
            return self.get_model(expr)
            # calling a function of a model
        if expr[0] in Comparision:
            return self.get_comparision(expr)
        raise NotImplementedError('Undefined expression')

    def get_comparision(self, compare):
        cond = compare[0]  # an string
        lexpr = compare[1]  # an list
        rexpr = compare[2]  # an list
        return self.get_expressions(lexpr) + cond + self.get_expressions(rexpr)

    def get_model(self, model):
        length = len(model[0])
        model_name = model[0][1:length + 1]
        return model_name + '.' + self.get_expressions(model[1])

    def get_limit(self, limit):
        pass

    def get_delete(self, delete_expr):
        pass

    def get_insert(self, insert):
        #temp = ['insert', 'Question.Answer', {'author': <Sym: arg_user>, 'text': <Sym: >, 'publish': <Sym: request_data_publish>, 'status': None, 'anonymous': <Sym: request_data_anonymous>, 'question': <Sym: ['getID', 'Question.Question', (), {'pk': <Sym: request_data_question>}, ['all', 'Question.Question']]>}]
        function_name = insert[0]
        target = insert[1]
        args = insert[2] # dictionary
        ir = 'insert ' + target + '('
        for arg in args.items():
            arg_name = arg[0]
            target_name = _make_immutable(arg[1])
            ir += arg_name + '=' + target_name + ','
        ir += ')\n'
        return ir

    def get_save(self, save):
        pass

    def get_link(self, link):
        pass

    def get_filter(self, filter):
        filter = _make_immutable(filter)
        ir = 'filter('
        args = filter[1]
        kwargs = filter[2]
        operations = filter[3]
        if len(args) > 0:
            # TODO: Meaning of args?
            ir += ','

        if len(kwargs) > 0:
            for kwarg in kwargs:
                ir += str(kwarg[0]) + '=' + str(kwarg[1][0])
            ir += ','

        if len(operations) > 0:
            ir += self.get_expressions(operations)

        ir += ')'
        return ir

    def get_exclude(self, exclude):
        pass

    def get_deref(self, deref):
        pass

    def get_all(self, all):
        field = all[1]
        return 'all(' + str(field) + ')'

    def get_exists(self, exists):
        expr = exists[1]
        return 'exists(' + self.get_expressions(expr) + ')'

    def get_first(self, first):
        pass

    def get_orderby(self, orderby):
        pass

    def get_singleton(self, singleton):
        pass

    def get_project(self, project):
        pass

    def get_field(self, field):
        pass

    def get_follow(self, follow):
        pass

    def get_get(self, get):
        get = _make_immutable(get)
        print('[DBG] get is ', get)
        ir = 'get('
        args = get[1]
        kwargs = get[2]
        operation = get[3]
        if len(args) > 0:
            # TODO: Meaning of args?
            ir += ','

        if len(kwargs) > 0:
            kwargs = _make_immutable(kwargs)
            print('[DBG]', kwargs)
            for kwarg in kwargs:
                if isinstance(kwarg[1], list):
                    ir += str(kwarg[0]) + '=' + str(kwarg[1][0])
                else:
                    ir += str(kwarg[0]) + '=' + str(kwarg[1])
            ir += ','

        if len(operation) > 0:
            ir += self.get_expressions(operation)

        ir += ')'
        return ir

    def get_cond(self, cond):
        # ('exists', ('filter', (), (('user', ('arg_user',)),), ('all', 'AccountOperation.UserFav')))
        if len(cond) < 1:
            return ''
        funccall = cond[0]
        expr = ""
        expr += str(funccall)
        expr += "("
        for arg in cond:
            if arg == funccall:
                continue
            else:
                print('in_cond ', arg)
                if type(arg) == tuple:
                    expr += self.get_expressions(arg)
            expr += ','
        expr += ")"
        return expr

    def get_effect(self, effect):
        if len(effect) < 1:
            return ''
        funccall = effect[0]
        expr = ''
        expr += str(funccall)
        expr += '('
        for arg in effect:
            if arg == funccall:
                continue
            if type(arg) == list:
                expr += self.get_expressions(arg)
            expr += ','
        expr += ')'
        return expr

    def printer(self):
        print (f'*** Intermediate Representation ** ', file=self.file)
        free_vars = self.path.free_vars
        path_conds = self.path.path_conds
        effects = self.path.effects
        """
        Transform vars, conds and effect to Low-level IR
        """
        ir = 'op unnamed('
        for val in free_vars.items():
            ir += str(val[0]) + ':' + str(val[1]) + ','
        ir += '){\n'
        for cond in path_conds:
            ir += 'guard' + self.get_cond(cond) + '\n'
            pass
        for effect in effects:
            ir += self.get_effect(effect) + '\n'
        ir += '}'
        print(ir, file=self.file)

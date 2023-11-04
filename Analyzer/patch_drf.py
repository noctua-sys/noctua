from Analyzer.symbolic import Sym
from Analyzer import notify
from Analyzer.patch_django import FakeUser
from rest_framework import request
from _collections_abc import Mapping
from Verifier.checker import *
import z3

from django.contrib.auth import get_user_model


class Data(Mapping):
    def __init__(self):
        self._readset = dict()

    def __iter__(self):
        raise ValueError('Symbolic request data cannot be iterated')

    def __len__(self):
        raise ValueError('Symbolic request data cannot be sized')

    def __getitem__(self, item):
        if item in self._readset:
            return self._readset[item]
        argname = "request_data_" + item
        type = int
        t = Type.INT()
        # FIXME: APP-SPECIFIC
        if argname.endswith('topic') or argname.endswith('_question'):
            pass
        elif argname.endswith('anonymous') or argname.endswith('accepted') or argname.endswith('favorited') or argname.endswith('hidden') or argname.endswith('public'):
            type = bool
            t = Type.BOOL()
        elif argname.endswith('text') or argname.endswith('question') or argname.endswith('status') or argname.endswith('topic_type') or argname.endswith('name') or argname.endswith('desc') or argname.endswith('parent_topic') or argname.endswith('body') or argname.endswith('topic') or argname.endswith('title') or argname.endswith('password') or argname.endswith('code') or argname.endswith('vote_type') or argname.endswith('nom') or argname.endswith('niveau') or argname.endswith('sujet') or argname.endswith('message') or argname.endswith('description') or argname.endswith('titre') or argname.endswith('intitulerPostGrade') or argname.endswith('intitulerSujet') or argname.endswith('diplomeGraduation') or argname.endswith('Encadreur') or argname.endswith('argument') or argname.endswith('gradeVoulu') or argname.endswith('title') or argname.endswith('hash') or argname.endswith('lieu_naissance') or argname.endswith('nationaliter') or argname.endswith('email') or argname.endswith('sexe') or argname.endswith('addresse') or argname.endswith('grade') or argname.endswith('mobile'):
            type = str
            t = Type.STR()

        o = Sym(notify.obtain_free_expr(argname, t), type)
        self._readset[item] = o
        return o

    def __delitem__(self, item):
        pass


class Request(request.Request):
    def __init__(self, *args, **kwargs):
        request.OldRequest.__init__(self, *args, **kwargs)
        self._data = Data()

    @property
    def user(self):
        if hasattr(self, '_user'):
            return self._user
        self._user = FakeUser()
        return self._user

    @property
    def data(self):
        return self._data

# 1. symbolic values are always associated with an expression.
# 2. free occurrences of names are understood as arguments.

from datetime import datetime
from Analyzer.notify import *
from Analyzer.utils import related_key_to_related_model, related_key_to_relation, simple_field_to_type
from Verifier.checker import *
import logging


PY_type = type


# Simply wrap expressions
class Sym(object):
    def __init__(self, expr, type: 'type', /, attrs={}):
        """
        @type is a concrete Python type.  If the Sym object is an Object, type should be the model class.
        """
        assert isinstance(expr, Expr)
        assert type is not None
        self.expr = expr
        self.type = type

        # Special attributes:
        # - _bool_expr
        # - _annotations: inherited from QuerySet.annotate
        for k, v in attrs.items():
            setattr(self, k, v)

    def __bool__(self):
        path = get_current_path()
        # _bool_expr can be used to override __bool__
        # it is set by first(), last(), etc.
        expr = self.__dict__.get('_bool_expr', self.expr)
        val = path.determine_bool(expr)
        if val:
            add_cond(expr)
        else:
            add_cond(Unary('not', expr))
        return val

    def translate(self):
        return self.expr

    def __repr__(self):
        return '<Sym: {}>'.format(self.expr)

    def __len__(self):
        # NOTE: return an unknown length
        return Sym(W(z3.FreshConst(z3.IntSort(), 'sym_len')), int)

    def __add__(self, other):
        other = _ensure_sym(other)
        type = None
        if self.type is int and other.type is int:
            type = int
        elif self.type is float and other.type is int:
            type = float
        elif self.type is int and other.type is float:
            type = float
        elif self.type is float and other.type is float:
            type = float
        return Sym(Binary('+', self.expr, other.expr), type)

    def __sub__(self, other):
        other = _ensure_sym(other)
        type = None
        if self.type is int and other.type is int:
            type = int
        elif self.type is float and other.type is int:
            type = float
        elif self.type is int and other.type is float:
            type = float
        elif self.type is float and other.type is float:
            type = float
        return Sym(Binary('-', self.expr, other.expr), type)

    def __mul__(self, other):
        other = _ensure_sym(other)
        type = None
        if self.type is int and other.type is int:
            type = int
        elif self.type is float and other.type is int:
            type = float
        elif self.type is int and other.type is float:
            type = float
        elif self.type is float and other.type is float:
            type = float
        return Sym(Binary('*', self.expr, other.expr), type)

    def __truediv__(self, other):
        other = _ensure_sym(other)
        type = None
        if (self.type is int or self.type is float) and (other.type is int or other.type is float):
            type = float
        return Sym(Binary('/', self.expr, other.expr), type)

    def __floordiv__(self, other):
        other = _ensure_sym(other)
        type = None
        if self.type is int and other.type is int:
            type = int
        elif self.type is float and other.type is int:
            type = float
        elif self.type is int and other.type is float:
            type = float
        elif self.type is float and other.type is float:
            type = float
        return Sym(Binary('//', self.expr, other.expr), type)

    def __lt__(self, other):
        other = _ensure_sym(other)
        return Sym(Binary('<', self.expr, other.expr), bool)

    def __gt__(self, other):
        other = _ensure_sym(other)
        return Sym(Binary('>', self.expr, other.expr), bool)

    def __eq__(self, other):
        if other is None:
            return False
        elif other == '':
            return False  # Assume strings are not empty. FIXME: check this assumption.
        other = _ensure_sym(other)
        return Sym(Binary('==', self.expr, other.expr), bool)

    def __ge__(self, other):
        other = _ensure_sym(other)
        return Sym(Binary('>=', self.expr, other.expr), bool)

    def __le__(self, other):
        other = _ensure_sym(other)
        return Sym(Binary('<=', self.expr, other.expr), bool)

    def __float__(self):
        return SymFloat(Unary('tofloat', self.expr))

    def __int__(self):
        return SymInt(Unary('toint', self.expr))

    def __str__(self):
        # FIXME(Sym): Should check the type of expr
        assert isinstance(self.expr, Expr)
        return SymStr(Unary('tostr', self.expr))

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # print('__setattr__', type(self), __name, __value)

        if 'type' not in self.__dict__:
            return

        # Track changed fields, so .save() can know which fields are changed.
        from django.db import models

        self_type = self.__dict__['type']
        if self_type is not None and issubclass(self_type, models.Model):
            model_name = self_type._meta.label
            try:
                fld = self_type._meta.get_field(name)
                simple_ty = simple_field_to_type(fld)
                value = _ensure_sym(value)

                if simple_ty:  # name denotes a field
                    # Cannot use setattr directly here because of recursion.
                    self.__dict__['expr'] = SetF(model_name, name, value.expr, self.expr)
                else:
                    relation_name, dir = related_key_to_relation(self_type, name)
                    if isinstance(fld, models.ForeignKey) and dir == 'forward':
                        add_effect(LinkObj(relation_name, self.expr, _ensure_sym(value).expr))
                    else:
                        print('TODO: should handle relation set, fld =', fld, type(fld), ', value =', value.expr, ', self =', self.expr)
            except models.FieldDoesNotExist:
                # Not a field nor a relation. Do nothing.
                pass

    def __getattr__(self, attr):
        """
        Branch the logic based on self.type.
        The most important job here is to maintain
        the type of the new Sym.
        """
        from django.db import models
        from django.db.models.fields.related import ForeignKey, ManyToManyField, OneToOneField
        from Analyzer.patch_django import QuerySet

        if hasattr(self, 'type') and self.type is not None and issubclass(self.type, models.Model):
            model_cls = self.type
            model_name = self.type._meta.label

            try:
                field = model_cls._meta.get_field(attr)
                simple_ty = simple_field_to_type(field)

                if simple_ty:  # a simple field
                    o = Sym(GetF(model_name, field.name, self.expr), simple_ty.pytype())
                    setattr(self, attr, o)
                else:
                    this_model_name = model_cls._meta.label
                    that_model, is_obj = related_key_to_related_model(model_cls, attr)
                    that_model_name = that_model._meta.label
                    rel_name, direction = related_key_to_relation(model_cls, attr)
                    result_expr = Follow(rel_name, direction, Singleton(this_model_name, self.expr))
                    if is_obj:
                        o = Sym(Any(that_model_name, result_expr), that_model)
                    else:
                        o = QuerySet(that_model, _translated = result_expr)
                    # Cannot setattr here because QuerySet is not a Sym.

                assert o is not None
                return o

            except models.FieldDoesNotExist:
                # not a field.
                # Case 1: side effect?
                if attr in ['delete', 'create', 'get_or_create', 'update', 'save']:
                    def f(*args, **kwargs):
                        if len(args) != 0 or len(kwargs) != 0:
                            print(f'[WARN] side effect received args = {args}, kwargs = {kwargs}')
                        cmd = None
                        ret = None
                        if attr == 'delete':
                            cmd = Delete(model_name, Singleton(model_name, self.expr))
                        elif attr == 'create':
                            logging.warn("side effect CREATE")
                            cmd = Update(model_name, self.expr)
                        elif attr == 'get_or_create':
                            logging.warn("side effect GET_OR_CREATE")
                            cmd = Update(model_name, self.expr)
                        elif attr == 'update':
                            cmd = Update(model_name, self.expr)
                        elif attr == 'save':
                            cmd = Update(model_name, Singleton(model_name, self.expr))
                        if cmd:
                            # When f is called, record a side effect.
                            add_effect(cmd)
                        else:
                            print('FUCK TODO')
                    setattr(self, attr, f)
                    return f

                # Case 2: relation?
                if hasattr(model_cls, attr) and isinstance(getattr(model_cls, attr), models.fields.related_descriptors.ReverseManyToOneDescriptor):
                    descriptor = getattr(model_cls, attr)
                    from_model = descriptor.field.model._meta.label
                    to_model = model_name
                    field_name = descriptor.field.name
                    rel_name = '{}__{}__{}'.format(from_model, to_model, field_name)
                    res = QuerySet(descriptor.field.model, _translated = Follow(rel_name, 'backward', Singleton(model_name, self.expr)))
                    assert res is not None
                    return res

                # Case 3: 'pk'
                elif attr == 'pk':
                    res = ObjToRef(model_name, self.expr)
                    setattr(self, attr, res)
                    return res

                # Case 4: for restframework. Patch relations.RelatedField.get_attribute
                elif attr == 'serializable_value':
                    def f(*args, **kwargs):
                        raise AttributeError
                    return f

                elif hasattr(model_cls, attr):
                    print(f'fuck, ATTR {attr} NOT RECOGNIZED BUT IT IS IN MODEL {model_cls._meta.label}')

                    # If model_cls.attr is callable, then it is a method.
                    value = getattr(model_cls, attr)
                    if callable(value):
                        def f(*args, **kwargs):
                            return value(self, *args, **kwargs)
                        setattr(self, attr, f)
                        return f

                    return None

                else:
                    raise RuntimeError(f'Attribute does not exist: {attr}')

        # ImageField handling
        elif self.type is str and attr == 'url':
            # ImageField has a 'url' field.  Here we just return itself.
            # NOTE(ImageField): we think of an ImageField as a URL to the image data.
            return self
        elif self.type is str and attr in ['height', 'width']:
            return 800

        # Retrieve annotations
        elif '_annotations' in self.__dict__ and attr in self._annotations:
            return self._annotations[attr](self)

        # This symbolic value is a Python value, yet we don't know
        # what that attribute is.
        elif self.type in [str, int, float, datetime]:
            raise AttributeError('Trying to get attr {} from symbolic value for python type {}, expr = {}'.format(attr, self.type, self.expr.cached_pprint(0)))

        elif self.type is not None:
            raise NotImplementedError('Trying to get attr {} from symbolic value for python type {}, expr = {}'.format(attr, self.type, self.expr.cached_pprint(0)))

        else:
            raise NotImplementedError("Know nothing about attr %s for type %s %s" % (attr, PY_type(self), self.expr.cached_pprint(0)))


class SymPython(Sym):
    """
    Any subclass of this class: `expr` MAY be a literal Python value.
    """
    pass


class SymInt(int, SymPython, Sym):
    def __new__(cls, *args, **kwargs):
        instance = int.__new__(cls, 0)
        return instance

    def __init__(self, expr):
        assert isinstance(expr, Expr)
        Sym.__init__(self, expr, int)

    def __str__(self):
        return Sym.__str__(self)

    def __repr__(self):
        return Sym.__repr__(self)


class SymBool(SymPython, Sym):
    def __init__(self, expr):
        Sym.__init__(self, expr, bool)

    def __bool__(self):
        if isinstance(self.expr, bool):
            return self.expr
        else:
            return super().__bool__()


class SymFloat(float, SymPython, Sym):
    def __new__(cls, *args, **kwargs):
        instance = float.__new__(cls, 0.0)
        return instance

    def __init__(self, expr):
        Sym.__init__(self, expr, float)

    def __bool__(self) -> bool:
        if isinstance(self.expr, float):
            return self.expr.__bool__()
        else:
            return super().__bool__()

    def __str__(self):
        return Sym.__str__(self)

    def __repr__(self):
        return Sym.__repr__(self)


class SymStr(str, SymPython, Sym):
    def __new__(cls, *args, **kwargs):
        instance = str.__new__(cls, '')
        return instance

    def __init__(self, expr):
        assert isinstance(expr, (Expr, Sym))
        if isinstance(expr, Sym):
            expr = expr.expr
        Sym.__init__(self, expr, str)

    def __bool__(self) -> bool:
        if isinstance(self.expr, str):
            if self.expr: return True
            else: return False
        else:
            return super().__bool__()

    def __str__(self):
        return self

    def __repr__(self):
        return Sym.__repr__(self)

    def trim(self):
        # TODO
        return self
        # return SymStr(['trim', self.expr])


class SymNull(SymPython, Sym):
    def __init__(self):
        Sym.__init__(self, None, None)

    def __bool__(self):
        return False


class SymList(SymPython, Sym):
    def __init__(self, expr: Expr):
        assert isinstance(expr, Expr)
        Sym.__init__(self, expr, list)

    def __bool__(self):
        # FIXME(ksqsf): cannot tell concrete lists from symbolic expressions...
        return super().__bool__()


def _ensure_sym(obj):
    """
    If @obj is a constant object, wrap it in a Sym object.
    """
    if obj is None:
        return SymNull()
    elif isinstance(obj, Sym):
        return obj
    elif isinstance(obj, bool):
        return SymBool(Bool(obj))
    elif isinstance(obj, int):
        return SymInt(Int(obj))
    elif isinstance(obj, str):
        return SymStr(Str(obj))
    elif isinstance(obj, float):
        return SymFloat(Real(obj))
    elif isinstance(obj, datetime):
        return Sym(Int(int(obj.timestamp())), datetime)
    else:
        raise RuntimeError('Unsupported object type in _ensure_sym: {}'.format(str(type(obj))))

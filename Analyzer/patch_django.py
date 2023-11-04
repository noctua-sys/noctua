from datetime import datetime
from Analyzer.utils import related_key_to_relation, simple_field_to_type
from django.db.models import query
import django.db.models as models
from Analyzer.symbolic import Sym, SymBool, SymFloat, SymInt, SymStr, _ensure_sym
from Analyzer import notify
from Verifier.checker import *
import z3


class QuerySet(query.OldQuerySet):
    def __init__(self, *args, **kwargs):
        translated_overriden = False
        if '_translated' in kwargs:
            translated_overriden = True
            self._translated = kwargs.pop('_translated')

        super().__init__(*args, **kwargs)
        self._label = self.model._meta.label
        self._iterable_class = None  # Make sure no DB calls are made
        self._meta = self.model._meta  # NOTE(ksqsf): pass _meta for get_field_info()
        self.query = None
        if not translated_overriden:
            self._translated = All(self.model._meta.label)
        self._annotations = dict()
        self._transform_obj = lambda x: x

    def _clone(self):
        c = self.__class__(model=self.model, query=None, using=self._db, hints=self._hints)
        c._sticky_filter = self._sticky_filter
        c._for_write = self._for_write
        c._prefetch_related_lookups = self._prefetch_related_lookups[:]
        c._known_related_objects = self._known_related_objects
        c._iterable_class = None
        c._fields = self._fields
        c._meta = self._meta
        c._annotations = self._annotations
        return c

    def __deepcopy__(self, memo):
        return self._clone()

    def new_from(self, translation: Expr):
        c = self._clone()
        c._translated = translation
        return c

    def _fetch_all(self):
        assert False

    def __getitem__(self, k):
        if isinstance(k, slice):
            # FIXME: k.start
            qs = self.new_from(Limit(self._label, k.stop, self.translate()))
            assert k.step is None
            return qs
        elif isinstance(k, int):
            obj = Sym(Any(self._label, Limit(self._label, 1, self.translate())),
                      self.model, attrs={'_meta': self._meta,
                                         '_annotations': self._annotations})
            return obj
        else:
            raise ValueError('unrecognized getitem arg: ' + str(k))

    def __bool__(self):
        trigger = self.exists()
        if trigger:
            return True
        else:
            return False

    def __len__(self):
        return self.count()

    def __iter__(self):
        # NOTE: We just return 2 objects
        attrs = {'_meta': self._meta,
                 '_annotations': self._annotations}
        o1 = Sym(Any(self._label, self.translate()), self.model, attrs=attrs)
        # o2 = Sym(Any(self._label, self.translate()), self.model, attrs=attrs)
        return iter([
            self._transform_obj(o1),
            # self._transform_obj(o2),
        ])

    def __repr__(self):
        return f'<QuerySet {self.translate()}>'

    def translate(self):
        """Returns the translation from QuerySet to the S-expression representation of SOIR."""
        assert isinstance(self._translated, Expr)
        return self._translated

    def __and__(self, other):
        raise NotImplementedError()

    def __or__(self, other):
        raise NotImplementedError()

    def aggregate(self, *args, **kwargs):
        raise NotImplementedError()

    def count(self):
        return SymInt(Count(self._label, self.translate()))

    def __resolve_filter_to_expr(self, reverse: bool, *args, **kwargs) -> Tuple[Expr, bool]:
        """
        Resolve filter as used in filter() and get() to Expr.
        Returns (expr, object?). If object? = True, then the type of Expr is object, otherwise it's set.
        """
        return self.__resolve_filter_to_expr1(self._label, self.translate(), self._meta, reverse, *args, **kwargs)

    @staticmethod
    def __resolve_filter_to_expr1(model_name, qs_expr, meta, reverse: bool, *args, **kwargs) -> Tuple[Expr, bool]:
        if 'pk' in kwargs:
            pk = kwargs['pk']
            return Deref(model_name, qs_expr, pk.expr), True

        elif len(args) == 1:
            pk = args[0]
            return Deref(model_name, qs_expr, pk.expr), True

        elif len(args) == 0 and len(kwargs) > 0:
            result_expr = qs_expr

            for key, value in kwargs.items():
                # FIXME: exclude(field=None), do nothing.
                if value == None:
                    continue

                key_parts = key.split('__')
                comparator = 'eq'
                field_name = key_parts[0]

                # Replace 'exact' by 'eq'
                if key_parts[-1] == 'exact':
                    key_parts[-1] = 'eq'

                if len(key_parts) > 2:
                    raise NotImplementedError('currently only support field__comp form, not: ' + str(key_parts))
                elif len(key_parts) == 2:
                    comparator = key_parts[1]
                f = meta.get_field(field_name)
                simple_ty = simple_field_to_type(f)

                if simple_ty:
                    # Getting simple_ty succeeded. It is indeed a simple field.
                    result_expr = Filteq(model_name, field_name, _ensure_sym(value).expr, result_expr, comparator, reverse)

                else:
                    # Otherwise, this field must be a related field.

                    # from django.db.models.fields.related_descriptors import ForwardManyToOneDescriptor, \
                    #     ReverseManyToOneDescriptor, ForwardOneToOneDescriptor, \
                    #     ReverseOneToOneDescriptor, ForwardManyToManyDescriptor, \
                    #     ReverseManyToManyDescriptor

                    components = key.split('__')

                    # FIXME: just support the easy case.
                    if len(components) > 2:
                        raise NotImplementedError('QuerySet filter resolution #components > 2: {}'.format(str(components)))

                    rel_name, _ = related_key_to_relation(meta.model, components[0])
                    rhs = None
                    if len(components) == 1:
                        if value.type == int:
                            # filter(related=id)
                            rhs = Int(value)
                        else:
                            # filter(related=obj)
                            rhs = value.id.expr
                        components.append('id')
                    elif len(components) == 2 and components[1] == 'id':
                        rhs = value.expr
                    else:
                        raise RuntimeError('Should not reach here for components ' + str(components))

                    # Since then, c[0] is relkey, c[1] is the remote field.

                    result_expr = FilteqRel(rel_name, components[1], rhs, result_expr, reverse)

            return result_expr, False

        else:
            raise NotImplementedError('get() get both args and kwargs {} {}'.format(str(args), str(kwargs)))

    def get(self, *args, **kwargs):
        expr, is_object = self.__resolve_filter_to_expr(False, *args, **kwargs)
        if not is_object:
            obj = Sym(Any(self._label, expr), self.model, attrs={'_meta': self._meta, '_annotations': self._annotations})
            notify.add_cond(Exists(self._label, ObjToRef(self._label, obj.expr)))
            #notify.add_cond(Unary('not', Empty(self._label, expr)))
            return obj
        else:
            obj = Sym(expr, self.model, attrs={'_meta': self._meta, '_annotations': self._annotations})
            notify.add_cond(Exists(self._label, ObjToRef(self._label, obj.expr)))
            return obj

    def create(self, **kwargs):
        from django.contrib.postgres.fields.jsonb import JSONField

        obj_expr: FreeObj = notify.obtain_free_obj_expr(self._label)
        obj = Sym(obj_expr, self.model, attrs={'_meta': self._meta, '_annotations': self._annotations})
        notify.add_cond(Unary('not', Exists(self._label, ObjToRef(self._label, obj.expr))))

        # Deferred links.
        to_link = {}

        unique_simple = set()
        unique_relation = set()
        unique_together = getattr(self._meta, 'unique_together', None)
        if unique_together:
            unique_together = list(iter(*unique_together))

        for field in self._meta.get_fields():
            # Record if this field is unique.
            try:
                if field.unique:
                    simple_ty = simple_field_to_type(field)
                    if simple_ty:
                        unique_simple.add(field.name)
                    else:
                        unique_relation.add(field.name)

            except AttributeError:
                pass

            if field.name in kwargs and kwargs[field.name] is not None:
                # Collect objects that should be linked to this new object.
                if isinstance(field, (models.ForeignKey, models.ManyToOneRel,
                                      models.OneToOneField, models.ManyToManyField,
                                      models.ManyToManyRel)):
                    to_link[field.name] = kwargs[field.name]
                else:
                    setattr(obj, field.name, _ensure_sym(kwargs[field.name]))

            else:
                # Construct a default value.
                if isinstance(field, (models.DateTimeField, models.DateField)):
                    if field.auto_now_add or field.auto_now or field.default is not None:
                        if field.default is not None:
                            print(f'[WARN] Assume {field} default is __now')
                        val = Sym(notify.obtain_free_expr('__now', Type.INT()), datetime)
                        setattr(obj, field.name, val)
                    elif field.default == models.NOT_PROVIDED:
                        raise RuntimeError(f'Field {field} without default values and supplied values')
                    else:
                        print('!!! Cannot understand concrete date values:', field.default())

                elif isinstance(field, models.AutoField):
                    name = '{}_{}'.format(obj_expr.name, field.name)
                    expr = notify.obtain_free_expr(name, Type.INT())
                    val = SymInt(expr)
                    if field.primary_key:
                        notify.add_cond(Binary('==', ObjToRef(self._label, obj.expr), ToRef(self._label, expr)))
                        notify.add_unique_id(name, self._label)
                    setattr(obj, field.name, val)

                elif isinstance(field, models.IntegerField):
                    if field.default != models.NOT_PROVIDED:
                        val = SymInt(Int(field.default))
                    else:
                        val = SymInt(Int(0))
                    setattr(obj, field.name, val)

                elif isinstance(field, (models.SlugField, models.CharField, models.TextField)):
                    if field.default != models.NOT_PROVIDED:
                        val = SymStr(Str(field.default))
                    else:
                        val = SymStr(Str(''))
                    setattr(obj, field.name, val)

                elif isinstance(field, (models.FloatField)):
                    if field.default != models.NOT_PROVIDED:
                        val = SymFloat(Real(field.default))
                    else:
                        val = SymFloat(Real(0))
                    setattr(obj, field.name, val)

                elif isinstance(field, models.BooleanField):
                    if field.default != models.NOT_PROVIDED:
                        assert isinstance(field.default, bool)
                        val = SymBool(Bool(field.default))
                    elif field.null:
                        val = SymBool(True)  # FIXME: Only for temporary debugging purposes.
                    else:
                        raise RuntimeError(f'Field {field} without default values and supplied values')
                    setattr(obj, field.name, val)

                elif isinstance(field, (models.ForeignKey,
                                        models.ManyToOneRel,
                                        models.OneToOneField,
                                        models.ManyToManyField,
                                        models.ManyToManyRel)):
                    # Do nothing. It should be fine, because related
                    # fields are handled by Sym.__getattr__.
                    pass

                elif isinstance(field, models.ImageField):
                    # Do nothing.  This makes this field "unknown".
                    pass

                elif isinstance(field, JSONField):
                    # Do nothing.  This makes this field "unknown".
                    pass

                else:
                    print('!!! Unknown field type in insert:', field, type(field))

        # Save this object *AFTER* the default values are set.
        notify.add_effect(Update(self._label, Singleton(self._label, obj.expr)))

        # Link obj with arg in the specified relation. We do this *after*
        # setting default values because ID needs to be initialized.
        for field_name, field_value in to_link.items():
            setattr(obj, field_name, field_value)

        # Now, issue Unique constraints.
        for field_name in unique_simple:
            if field_name == 'id':
                # Skip 'id' because it is already handled above.
                continue
            unique_constraint = Empty(self._label, Filteq(self._label, field_name, getattr(obj, field_name).expr, All(self._label)))
            # print('UNIQUE', unique_constraint.cached_pprint(0))
            notify.add_cond(unique_constraint)
        # ForeignKey can have unique=True.
        # Conservatively handle it, ie. do nothing.
        if len(unique_relation) > 0:
            print(f'[WARN] unique_relation found in model {self._label} :', unique_relation)
        # assert len(unique_relation) == 0  
        if unique_together:
            qs = self.new_from(All(self._label))
            # Translate (f1, f2) into filter(f2=v2, filter(f1=v1)).empty()
            qs, is_obj = qs.__resolve_filter_to_expr(False, **{each: getattr(obj, each) for each in unique_together})
            assert not is_obj
            unique_constraint = Empty(self._label, qs)
            notify.add_cond(unique_constraint)

        return obj

    def bulk_create(self, objs, batch_size=None):
        raise NotImplemented
        ret = []
        for obj in objs:
            ret.append(self.create(obj))
        return ret

    def get_or_create(self, defaults=None, **kwargs):
        expr, is_object = self.__resolve_filter_to_expr(False, **kwargs)
        if not is_object:
            existence = SymBool(Unary('not', Empty(self._label, expr)))
            obj = Sym(Any(self._label, expr), self.model, attrs={'_meta': self._meta, '_annotations': self._annotations})
        else:
            obj = Sym(expr, self.model, attrs={'_meta': self._meta, '_annotations': self._annotations})
            existence = SymBool(Exists(self._label, ObjToRef(self._label, expr)))
        # branching depending on whether this object exists.
        # this is necessitated by unique constraints.
        if existence:
            return obj
        else:
            return self.create(**kwargs)

    def update_or_create(self, defaults=None, **kwargs):
        # TODO
        raise NotImplementedError()

    def earliest(self, *fields, field_name=None):
        # TODO
        raise NotImplementedError()

    def latest(self, *fields, field_name=None):
        # TODO
        raise NotImplementedError()

    def first(self):
        # FIXME: order
        return Sym(Any(self._label, Limit(self._label, 1, self.translate())), self.model,
                   attrs={'_meta': self._meta,
                          '_bool_expr': Unary('not', Empty(self._label, self.translate())),
                          '_annotations': self._annotations})

    def last(self):
        # FIXME: order
        return Sym(Limit(self._label, 1, self.translate()), self.model,
                   attrs={'_meta': self._meta,
                          '_bool_expr': Unary('not', Empty(self._label, self.translate())),
                          '_annotations': self._annotations})

    def delete(self):
        notify.add_effect(Delete(self._label, self.translate()))
        return SymInt(W(z3.FreshConst(z3.IntSort(), 'django_delete')))  # unknown number of deleted rows

    def update(self):
        notify.add_effect(Update(self._label, self.translate()))
        return SymInt(W(z3.FreshConst(z3.IntSort(), 'django_update')))  # unknown number of updated rows

    def exists(self):
        """
        Semantics: This queryset is not empty.
        """
        return Sym(Unary('not', Empty(self._label, self.translate())), bool)

    def all(self):
        return self.new_from(self.translate())

    def filter(self, *args, **kwargs):
        result_expr, is_object = self.__resolve_filter_to_expr(False, *args, **kwargs)
        if is_object:
            return self.new_from(Singleton(self._label, result_expr))
        else:
            return self.new_from(result_expr)

    def exclude(self, *args, **kwargs):
        '''Like filter(), but excludes selected results.'''
        result_expr, is_object = self.__resolve_filter_to_expr(True, *args, **kwargs)
        if is_object:
            return self.new_from(Singleton(self._label, result_expr))
        else:
            return self.new_from(result_expr)

    def union(self, *other_qs, all=False):
        cur = self
        for qs in other_qs:
            cur = cur._union(qs)
        return cur

    def intersection(self, *other_qs):
        # ??
        raise NotImplementedError()

    def difference(self, *other_qs):
        # ??
        raise NotImplementedError()

    def select_for_update(self, nowait=False, skip_locked=False, of=()):
        raise NotImplementedError()

    def select_related(self, *fields):
        print('select_related')
        raise NotImplementedError()

    def prefetch_related(self, *fields):
        # prefetch_related is just a performance hack.
        return self

    def only(self, *args, **kwargs):
        return self

    def defer(self, *args, **kwargs):
        return self

    def annotate(self, *args, **kwargs):
        annotations = dict()
        for key, agg in kwargs.items():
            if isinstance(agg, models.Count):
                source_exprs = agg.get_source_expressions()
                assert len(source_exprs) == 1, 'Not supported yet'
                field_name = source_exprs[0].name
                def annotation_interp(obj):
                    assert isinstance(obj, Sym)
                    qs = getattr(obj, field_name)
                    assert isinstance(qs, QuerySet)
                    return SymInt(qs.count())
                annotations[key] = annotation_interp
            else:
                raise NotImplementedError('only support Count annotation')

        qs = self._clone()
        qs._annotations.update(annotations)
        return qs

    def order_by(self, *field_names):
        qs = self
        for f in field_names:
            if f.startswith('-'):
                # FIXME: check if my interpretation of - is correct.
                qs = qs.new_from(OrderBy(self._label, f[1:], 'desc', qs.translate()))
            elif f.startswith('+'):
                qs = qs.new_from(OrderBy(self._label, f[1:], 'asc', qs.translate()))
            else:
                qs = qs.new_from(OrderBy(self._label, f, 'asc', qs.translate()))
        return qs

    def distinct(self, *field_names):
        print('distinct')
        raise NotImplementedError()
        # return self.new_from(['distinct', field_names, self.translate()])

    def reverse(self):
        print('reverse')
        raise NotImplementedError()
        # return self.new_from(['reverse', self.translate()])

    def _union(self, o):
        return self.new_from(Union(self._label, self.translate(), o.translate()))

    def values_list(self, *fields):
        ret = self._clone()
        def transform_obj_values_list(obj):
            return tuple([GetF(self._label, field, obj.expr) for field in fields])
        ret._transform_obj = transform_obj_values_list
        return ret


class FakeUser(Sym):
    """FakeUser is a symbolic value representing an active, regular (non-staff), and authenticated User."""
    def __init__(self, *args, **kwargs):
        from django.contrib.auth import get_user_model  # prevent circular

        self.is_authenticated = True
        self.is_active = True
        self.is_staff = False

        pk_expr = notify.obtain_free_expr('arg_user_id', Type.INT())
        self.pk = SymInt(pk_expr)
        self._bool_expr = Bool(True)  # Always assume the user exists
        user_model = get_user_model()
        user_model_name = user_model._meta.label
        super().__init__(
            Deref(user_model_name, All(user_model_name), ToRef(user_model_name, pk_expr)),
            user_model
        )

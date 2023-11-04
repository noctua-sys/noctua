from enum import Enum, auto
from adt import adt, Case
from typing import Callable, Dict, List, OrderedDict, Tuple, Optional, List
import z3
import logging
from copy import deepcopy
from Verifier.timeout import timedcall


_id = 0
def gen_id():
    global _id
    _id += 1
    return str(_id)


simplify_params = {
    'rewrite_patterns': True,
    'som': True,
    'sort_store': True,
    'elim_to_real': True
}

def simplify(z3expr):
    return z3.simplify(z3expr, **simplify_params)


def MergeModelSet(context: 'Context', mname: str, A: z3.ExprRef, B_prioritized: z3.ExprRef) -> z3.ExprRef:
    """
    Merge model sets A B. Prioritize elements in B.
    """
    B = B_prioritized   # for better documentation

    A_ids = context.set_ids[mname](A)
    B_ids = context.set_ids[mname](B)
    A_data = context.set_data[mname](A)
    B_data = context.set_data[mname](B)
    A_order = context.set_order[mname](A)
    B_order = context.set_order[mname](B)

    x = z3.FreshConst(context.ref_sort[mname], 'MergeModelSet_ref')

    res_ids = z3.SetUnion(A_ids, B_ids)
    res_data = z3.Lambda([x], z3.If(
        z3.IsMember(x, B_ids),
        z3.Select(B_data, x),
        z3.Select(A_data, x)
    ))
    res_order = z3.Lambda([x], z3.If(
        z3.IsMember(x, B_ids),
        z3.Select(B_order, x),
        z3.Select(A_order, x)
    ))
    return simplify(context.set_mk[mname](res_data, res_ids, res_order))


def ExcludeModelSet(context: 'Context', mname: str, A: z3.ExprRef, B: z3.ExprRef) -> z3.ExprRef:
    """
    A - B.
    """
    A_ids = context.set_ids[mname](A)
    B_ids = context.set_ids[mname](B)
    A_data = context.set_data[mname](A)
    A_order = context.set_order[mname](A)

    res_ids = z3.SetDifference(A_ids, B_ids)
    res_data = A_data
    res_order = A_order
    return simplify(context.set_mk[mname](res_data, res_ids, res_order))


class Attr(Enum):
    PRIMARY = auto()
    UNIQUE = auto()
    OPTIONAL = auto()

@adt
class Type:
    BOOL: Case # booleans
    INT: Case # arbitrary precision integers
    REAL: Case # arbitrary precision floats
    DATE: Case # datetime
    STR: Case # strings

    OBJ: Case[str]
    REF: Case[str]
    SET: Case[str]

    def __str__(self):
        return self.match(
            bool=lambda: 'bool',
            int=lambda: 'int',
            real=lambda: 'real',
            date=lambda: 'date',
            str=lambda: 'str',
            obj=lambda x: f'obj[{x}]',
            ref=lambda x: f'ref[{x}]',
            set=lambda x: f'set[{x}]',
        )

    def __repr__(self):
        return str(self)

    def smt(self, context) -> z3.Sort:
        assert not context or isinstance(context, Context)
        return self.match(
            bool=lambda: z3.BoolSort(),
            int=lambda: z3.IntSort(),
            real=lambda: z3.RealSort(),
            str=lambda: z3.StringSort(),
            date=lambda: z3.IntSort(),
            obj=lambda model: context.obj_sort[model],
            set=lambda model: context.set_sort[model],
            ref=lambda model: context.ref_sort[model],
        )

    def pytype(self) -> type:
        from django.utils.datetime_safe import datetime
        def _raise_exc(exc):
            raise exc
        return self.match(
            bool=lambda: bool,
            int=lambda: int,
            real=lambda: float,
            str=lambda: str,
            date=lambda: datetime,
            obj=lambda model: _raise_exc(ValueError('pytype does not support obj')),
            set=lambda model: _raise_exc(ValueError('pytype does not support set')),
            ref=lambda model: _raise_exc(ValueError('pytype does not support ref')),
        )

    @staticmethod
    def from_pytype(typ: type) -> 'Type':
        """Construct a Type object from a Python type object."""
        assert isinstance(typ, type)
        assert typ in [int, float, str, bool]
        if typ is int:
            return Type.INT()
        elif typ is str:
            return Type.STR()
        elif typ is float:
            return Type.REAL()
        elif typ is bool:
            return Type.BOOL()
        else:
            raise ValueError('Unrecognized pytype: ' + str(typ))

    @property
    def model(self):
        def _raise_exc(exc):
            raise exc
        return self.match(
            bool=lambda: _raise_exc(ValueError('simple types do not have an associated model')),
            int=lambda: _raise_exc(ValueError('simple types do not have an associated model')),
            real=lambda: _raise_exc(ValueError('simple types do not have an associated model')),
            str=lambda: _raise_exc(ValueError('simple types do not have an associated model')),
            date=lambda: _raise_exc(ValueError('simple types do not have an associated model')),
            obj=lambda model: model,
            set=lambda model: model,
            ref=lambda model: model,
        )


class RelationKind(Enum):
    ONE_ONE = auto()
    MANY_ONE = auto()
    MANY_MANY = auto()


class Field:
    def __init__(self, name: str, ty, attrs):
        self.name = name
        self.ty = ty
        self.attrs = attrs


class Model:
    def __init__(self, name: str, fields: Optional[List[Field]]):
        self.name = name
        self.fields = OrderedDict({f.name: f for f in fields}) if fields else OrderedDict()
        self.field_to_idx: OrderedDict[str, int] = OrderedDict()
        self._update_field_indices()

    def __str__(self):
        return self.name

    def add_field(self, f: Field):
        self.fields[f.name] = f
        self._update_field_indices()

    def primary_fields(self) -> List[Field]:
        """Return all fields with attr PRIMARY."""
        ret = []
        for f in self.fields.values():
            if Attr.PRIMARY in f.attrs:
                ret.append(f)
        return ret

    def data_fields(self) -> List[Field]:
        """Return all fields that can be in an object."""
        ret = []
        for f in self.fields.values():
            ret.append(f)
        return ret

    def ref_sort(self):
        sorts = [f.ty.smt(None) for f in self.primary_fields()]
        return z3.TupleSort(self.name + "_ref", sorts)

    def obj_sort(self):
        # Use all fields in the object.
        sorts = [f.ty.smt(None) for f in self.data_fields()]
        return z3.TupleSort(self.name + "_obj", sorts)

    def order_sort(self):
        return z3.ArraySort(self.ref_sort()[0], z3.IntSort())

    def set_sort(self):
        data_sort = z3.ArraySort(self.ref_sort()[0], self.obj_sort()[0])
        ids_sort = z3.SetSort(self.ref_sort()[0])
        order_sort = self.order_sort()
        # NOTE(set): field order: data, ids, order
        ret = z3.TupleSort(self.name + '_set', [data_sort, ids_sort, order_sort])
        # print(ret)
        return ret

    def _update_field_indices(self):
        for i, field_name in enumerate(self.fields.keys()):
            self.field_to_idx[field_name] = i


class Relation:
    def __init__(self, name: str, kind: RelationKind, from_: Model, to_: Model):
        self.name = name
        self.kind = kind
        self.from_ = from_
        self.to_ = to_


class Op:
    """
    Op should be stateless!!
    """
    def __init__(self, name: str, argspecs: List[Tuple[str, Type]], body, unique_ids: Dict[str,str]):
        self.name = name
        self.argspecs = argspecs
        self.body = body
        self.unique_ids = unique_ids

    def generate_args(self, context: 'Context', prefix=''):
        ret = {}
        for (arg_name, arg_type) in self.argspecs:
            assert arg_name not in ret
            ret[arg_name] = z3.FreshConst(arg_type.smt(context), prefix + arg_name)
        return ret

    def generate_shadow(self, context: 'Context', prefix='') -> 'Shadow':
        """Generate a shadow operation."""
        args = self.generate_args(context, prefix)
        return Shadow(self, args, self.unique_ids)

    def effect(self, context: 'Context', sys: 'SystemState', args):
        raise NotImplementedError('An operation must override "effect".')

    def condition(self, context: 'Context', sys: 'SystemState', args) -> z3.BoolRef:
        raise NotImplementedError('An operation must override "condition".')


class Shadow:
    """A shadow refers to a closed effect with all of its arguments filled in.

    For convenience, a Semantics object is also embedded."""
    def __init__(self, gen_op, args: Dict[str, z3.ExprRef], unique_ids: Dict[str,str]):
        self.gen_op = gen_op
        self.args = args
        self.unique_ids = unique_ids

    def judge(self, context, sys) -> z3.ExprRef:
        context.set_free_env(self.args)
        res = self.gen_op.condition(context, sys, self.args)
        return res

    def apply(self, context, sys, prefix=None) -> z3.ExprRef:
        context.set_free_env(self.args)
        ret = self.gen_op.effect(context, sys, self.args, prefix='')
        # Generate dummy const for debugging.
        # The value will be displayed in the model.
        if prefix:
            for m in context.models.keys():
                set_sort = context.set_sort[m]
                x = z3.FreshConst(set_sort, prefix)
                context.solver.add(x == ret.models[m])
        return ret

    def smt_vars(self) -> List[z3.ExprRef]:
        return list(self.args.values())

    def smt_unique_vars(self) -> List[z3.ExprRef]:
        ret = []
        for name in self.unique_ids:
            ret.append(self.args[name])
        return ret



class SystemState:
    """A wrapper over (Model States, Relation States)."""
    def __init__(self, model_states: Dict[str, z3.ExprRef], relation_states: Dict[str, z3.ExprRef]):
        self.model_states = model_states
        self.relation_states = relation_states

    @property
    def models(self):
        return self.model_states

    @property
    def relations(self):
        return self.relation_states

    def smt_vars(self) -> List[z3.ExprRef]:
        return list(x for x in self.model_states.values()) + list(x for x in self.relation_states.values())


class Context:
    """Evaluation context."""
    def __init__(self, solver, models: OrderedDict[str, Model], relations: OrderedDict):
        self.solver = solver
        self.models = models
        self.relations: OrderedDict[str, Relation] = relations

        # Sorts and constructors for complex data types.
        self.ref_sort: OrderedDict[str, z3.Sort] = OrderedDict()
        self.ref_mk = OrderedDict()
        self.ref_projs = OrderedDict()
        self.obj_sort: OrderedDict[str, z3.Sort] = OrderedDict()
        self.obj_mk = OrderedDict()
        self.obj_projs = OrderedDict()
        self.set_sort: OrderedDict[str, z3.Sort] = OrderedDict()
        self.set_mk = OrderedDict()
        self.set_ids = OrderedDict()
        self.set_data = OrderedDict()
        self.set_order = OrderedDict()

        self.optional = OrderedDict()
        self.pair = OrderedDict()
        self.oneone = OrderedDict()
        self.manyone = OrderedDict()
        self.manymany = OrderedDict()

        self.free_env: Dict[str, z3.ExprRef] = dict()

        for model in self.models.values():
            self.define_model(model)
        for relation in self.relations.values():
            self.define_relation(relation)

    def set_free_env(self, free_env: Dict[str, z3.ExprRef]):
        self.free_env = free_env

    def get_free_val(self, free_name: str):
        return self.free_env[free_name]

    def fresh(self, sort, prefix='c'):
        return z3.FreshConst(sort, prefix)

    def assign(self, const, value):
        self.solver.add(const == value)

    def fresh_assign(self, sort, value, prefix='c'):
        ret = self.fresh(sort, prefix)
        self.assign(sort, value)
        return ret

    def add(self, z3expr: z3.ExprRef):
        self.solver.add(simplify(z3expr))

    def get_or_define_optional(self, sort):
        if str(sort) in self.optional:
            return self.optional[str(sort)]
        Optional = z3.Datatype('Optional_' + str(sort))
        Optional.declare('just', ('fromJust', sort))
        Optional.declare('nothing')
        Optional = Optional.create()
        self.optional[str(sort)] = Optional
        return Optional

    def get_or_define_pair(self, sort1, sort2):
        id = (str(sort1), str(sort2))
        if id in self.pair:
            return self.pair[id]
        Pair = z3.Datatype('Pair_%s_%s' % id)
        Pair.declare('cons', ('car', sort1), ('cdr', sort2))
        Pair = Pair.create()
        self.pair[id] = Pair
        return Pair

    def define_model_ref(self, model: Model):
        (sort, mk, projs) = model.ref_sort()
        self.ref_sort[model.name] = sort
        self.ref_mk[model.name] = mk
        self.ref_projs[model.name] = {model.primary_fields()[i].name: proj for (i, proj) in enumerate(projs)}

    def define_model_obj(self, model: Model):
        (sort, mk, projs) = model.obj_sort()
        self.obj_sort[model.name] = sort
        self.obj_mk[model.name] = mk
        self.obj_projs[model.name] = {model.data_fields()[i].name: proj for (i, proj) in enumerate(projs)}

    def define_model_set(self, model: Model):
        (sort, mk, projs) = model.set_sort()
        self.set_sort[model.name] = sort
        self.set_mk[model.name] = mk
        self.set_data[model.name] = projs[0]
        self.set_ids[model.name] = projs[1]
        self.set_order[model.name] = projs[2]

    def define_model(self, model: Model):
        self.define_model_obj(model)
        self.define_model_ref(model)
        self.define_model_set(model)

    def define_relation_one_one(self, r: Relation):
        from_ref_sort = self.ref_sort[r.from_.name]
        to_ref_sort = self.ref_sort[r.to_.name]
        builder = z3.Datatype('OneOne_' + r.name)
        builder.declare('create',
                        ('getForward', z3.ArraySort(from_ref_sort, self.get_or_define_optional(to_ref_sort))),
                        ('getBackward', z3.ArraySort(to_ref_sort, self.get_or_define_optional(from_ref_sort))))
        self.oneone[r.name] = builder.create()

    def many_one_forward_sort(self, r: Relation):
        from_ref_sort = self.ref_sort[r.from_.name]
        to_ref_sort = self.ref_sort[r.to_.name]
        return z3.ArraySort(from_ref_sort, self.get_or_define_optional(to_ref_sort))

    def many_one_backward_sort(self, r: Relation):
        from_ref_sort = self.ref_sort[r.from_.name]
        to_ref_sort = self.ref_sort[r.to_.name]
        return z3.ArraySort(to_ref_sort, z3.SetSort(from_ref_sort))

    def define_relation_many_one(self, r: Relation):
        builder = z3.Datatype('ManyOne_' + r.name)
        builder.declare('create',
                        ('getForward', self.many_one_forward_sort(r)),
                        ('getBackward', self.many_one_backward_sort(r)),)
        self.manyone[r.name] = builder.create()

    def destruct_relation_state(self, r: Relation, rs: z3.ExprRef) -> Tuple[z3.ExprRef, z3.ExprRef]:
        '''Destruct a relation state object into forward and backward.'''
        if r.kind == RelationKind.ONE_ONE:
            return self.oneone[r.name].getForward(rs), self.oneone[r.name].getBackward(rs)
        elif r.kind == RelationKind.MANY_ONE:
            return self.manyone[r.name].getForward(rs), self.manyone[r.name].getBackward(rs)
        else:
            return self.manymany[r.name].getForward(rs), self.manymany[r.name].getBackward(rs)

    def many_many_forward_sort(self, r: Relation):
        from_ref_sort = self.ref_sort[r.from_.name]
        to_ref_sort = self.ref_sort[r.to_.name]
        return z3.ArraySort(from_ref_sort, z3.SetSort(to_ref_sort))

    def many_many_backward_sort(self, r: Relation):
        from_ref_sort = self.ref_sort[r.from_.name]
        to_ref_sort = self.ref_sort[r.to_.name]
        return z3.ArraySort(to_ref_sort, z3.SetSort(from_ref_sort))

    def define_relation_many_many(self, r: Relation):
        builder = z3.Datatype('ManyMany_' + r.name)
        builder.declare('create',
                        ('getForward', self.many_many_forward_sort(r)),
                        ('getBackward', self.many_many_backward_sort(r)))
        self.manymany[r.name] = builder.create()

    def define_relation(self, rel: Relation):
        if rel.kind == RelationKind.ONE_ONE:
            self.define_relation_one_one(rel)
        elif rel.kind == RelationKind.MANY_ONE:
            self.define_relation_many_one(rel)
        elif rel.kind == RelationKind.MANY_MANY:
            self.define_relation_many_many(rel)
        else:
            raise ValueError('Unknown relation kind: ' + str(rel.kind))

    def well_formed_model_state(self, model: str, qs: z3.ExprRef) -> z3.ExprRef:
        model = self.models[model]

        # WF1: object-id correspondence
        # forall r in S.ids, r == S.data[r].id
        data = self.set_data[model.name](qs)
        ids = self.set_ids[model.name](qs)
        order = self.set_order[model.name](qs)
        ref = z3.FreshConst(self.ref_sort[model.name], 'wf1_ref_')
        WF1 = z3.ForAll([ref], z3.Implies(z3.IsMember(ref, ids), (ref == self.obj_to_ref(model.name, z3.Select(data, ref)))))

        # WF2: uniqueness of unique fields
        clauses = []
        for field in model.fields.values():
            if Attr.UNIQUE in field.attrs:
                # Example:
                # forall ms ref1 ref2, ref1 in ms.ids and ref2 in ms.ids => ms.data[ref1].name == ms.data[ref2].name => ref1 = ref2
                r1 = z3.FreshConst(self.ref_sort[model.name], 'wf2_ref1_')
                r2 = z3.FreshConst(self.ref_sort[model.name], 'wf2_ref2_')
                o1 = z3.Select(data, r1)
                o2 = z3.Select(data, r2)
                o1f = self.obj_projs[model.name][field.name](o1)
                o2f = self.obj_projs[model.name][field.name](o2)
                # If o1 and o2 belong to some set, o1.f == o2.f implies o1 == o2.
                clauses.append(z3.ForAll([r1, r2], z3.Implies(
                    z3.And([z3.IsMember(r1, ids),
                            z3.IsMember(r2, ids)]),
                    z3.Implies(o1f == o2f, z3.And(r1 == r2, o1 == o2))
                )))
        WF2 = z3.And(clauses)

        # # WF3: uniqueness of order
        # r1 = z3.FreshConst(self.ref_sort[model.name], 'wf3_ref1')
        # r2 = z3.FreshConst(self.ref_sort[model.name], 'wf3_ref2')
        # o1 = z3.Select(data, r1)
        # o2 = z3.Select(data, r2)
        # WF3 = z3.ForAll([r1, r2], z3.Implies(
        #     z3.And([z3.IsMember(r1, ids),
        #             z3.IsMember(r2, ids)]),
        #     z3.Implies(z3.Select(order, r1) == z3.Select(order, r2),
        #                r1 == r2)
        # ))
        WF3 = True

        return z3.And([WF1, WF2, WF3])

    def well_formed_relation_state_oneone(self, rname: str, sys: SystemState) -> z3.ExprRef:
        r = self.relations[rname]
        assert r.kind == RelationKind.ONE_ONE

        from_name = r.from_.name
        to_name = r.to_.name
        from_ref_sort = self.ref_sort[from_name]
        to_ref_sort = self.ref_sort[to_name]

        optional_from_ref = self.get_or_define_optional(from_ref_sort)
        optional_to_ref = self.get_or_define_optional(to_ref_sort)
        oneone = self.oneone[rname]
        rstate = sys.relations[rname]
        from_state = sys.models[r.from_.name]
        to_state = sys.models[r.to_.name]

        ref1 = z3.FreshConst(from_ref_sort, 'wfr11_from_ref_')
        ref2 = z3.FreshConst(to_ref_sort, 'wfr11_to_ref_')

        return z3.And([
            z3.ForAll([ref1, ref2], z3.Implies((optional_to_ref.just(ref2) == z3.Select(oneone.getForward(rstate), ref1)),
                                               z3.And([z3.IsMember(ref1, self.set_ids[from_name](from_state)),
                                                       z3.IsMember(ref2, self.set_ids[to_name](to_state))]))),
            z3.ForAll([ref1, ref2],
                ((optional_to_ref.just(ref2) == z3.Select(oneone.getForward(rstate), ref1)))
                ==
                (optional_from_ref.just(ref1) == z3.Select(oneone.getBackward(rstate), ref2)))
        ])

    def well_formed_relation_state_manyone(self, rname: str, sys: SystemState) -> z3.ExprRef:
        r = self.relations[rname]
        assert r.kind == RelationKind.MANY_ONE

        from_name = r.from_.name
        to_name = r.to_.name
        from_ref_sort = self.ref_sort[from_name]
        to_ref_sort = self.ref_sort[to_name]
        optional_to_ref = self.get_or_define_optional(to_ref_sort)
        manyone = self.manyone[rname]
        rstate = sys.relations[rname]
        from_state = sys.models[from_name]
        to_state = sys.models[to_name]

        ref1 = z3.FreshConst(from_ref_sort, 'wfr31_from_ref_')
        ref2 = z3.FreshConst(to_ref_sort, 'wfr31_to_ref_')

        return z3.And([
            z3.ForAll([ref1, ref2], z3.Implies(optional_to_ref.just(ref2) == z3.Select(manyone.getForward(rstate), ref1),
                                               z3.And([z3.IsMember(ref1, self.set_ids[from_name](from_state)),
                                                       z3.IsMember(ref2, self.set_ids[to_name](to_state))]))),
            z3.ForAll([ref1, ref2], z3.Implies(
                (optional_to_ref.just(ref2) == z3.Select(manyone.getForward(rstate), ref1)),
                z3.IsMember(ref1, z3.Select(manyone.getBackward(rstate), ref2))
            )),
            z3.ForAll([ref1, ref2], z3.Implies(
                z3.IsMember(ref1, z3.Select(manyone.getBackward(rstate), ref2)),
                (optional_to_ref.just(ref2) == z3.Select(manyone.getForward(rstate), ref1))
            )),
        ])

    def well_formed_relation_state_manymany(self, rname: str, sys: SystemState) -> z3.ExprRef:
        r = self.relations[rname]
        assert r.kind == RelationKind.MANY_MANY

        from_name = r.from_.name
        to_name = r.to_.name
        from_ref_sort = self.ref_sort[from_name]
        to_ref_sort = self.ref_sort[to_name]
        manymany = self.manymany[rname]
        rstate = sys.relations[rname]
        from_state = sys.models[from_name]
        to_state = sys.models[to_name]

        ref1 = z3.FreshConst(from_ref_sort, 'wfr33_from_ref_')
        ref2 = z3.FreshConst(to_ref_sort, 'wfr33_to_ref_')

        return z3.And([
            z3.ForAll([ref1, ref2], z3.Implies((z3.IsMember(ref2, z3.Select(manymany.getForward(rstate), ref1))),
                                               z3.And([z3.IsMember(ref1, self.set_ids[from_name](from_state)),
                                                       z3.IsMember(ref2, self.set_ids[to_name](to_state))]))),
            z3.ForAll([ref1, ref2], (z3.IsMember(ref2, z3.Select(manymany.getForward(rstate), ref1))
                                    == z3.IsMember(ref1, z3.Select(manymany.getBackward(rstate), ref2)))),
        ])

    def well_formed_relation_state(self, rname: str, sys: SystemState) -> z3.ExprRef:
        r = self.relations[rname]
        if r.kind == RelationKind.MANY_MANY:
            return self.well_formed_relation_state_manymany(rname, sys)
        elif r.kind == RelationKind.MANY_ONE:
            return self.well_formed_relation_state_manyone(rname, sys)
        else:
            return self.well_formed_relation_state_oneone(rname, sys)

    def well_formed_system_state(self, sys: 'SystemState') -> z3.ExprRef:
        return z3.And([self.well_formed_model_state(model, sys.models[model]) for model in self.models.keys()] +
                      [self.well_formed_relation_state(rname, sys) for rname in self.relations.keys()])

    def sys_eq(self, sys1: 'SystemState', sys2: 'SystemState') -> z3.ExprRef:
        assert set(sys1.model_states.keys()) == set(sys2.model_states.keys())
        assert set(sys1.relation_states.keys()) == set(sys2.relation_states.keys())
        conjs = []

        # model equality, relaxed to only valid ids
        for model in sys1.model_states.keys():
            ref = z3.FreshConst(self.ref_sort[model], 'syseq_ref')
            ids1 = self.set_ids[model](sys1.model_states[model])
            ids2 = self.set_ids[model](sys2.model_states[model])
            data1 = self.set_data[model](sys1.model_states[model])
            data2 = self.set_data[model](sys2.model_states[model])
            conjs.append(
                z3.Or(
                    sys1.model_states[model] == sys2.model_states[model],  # maybe faster
                    # ids1 == ids2 /\ forall ref, ref in ids1 && ref in ids2 ==> data1[ref] = data2[ref]
                    z3.And([
                        (ids1 == ids2),
                        z3.ForAll([ref], z3.Implies(z3.And([z3.IsMember(ref, ids1),
                                                            z3.IsMember(ref, ids2)]),
                                                    z3.Select(data1, ref) == z3.Select(data2, ref)))
                    ])
            ))

        # relation equality
        for rname in sys1.relation_states.keys():
            relation = self.relations[rname]
            rstate1 = sys1.relation_states[rname]
            rstate2 = sys2.relation_states[rname]
            r1 = z3.FreshConst(self.ref_sort[relation.from_.name], 'syseq_r1')
            r2 = z3.FreshConst(self.ref_sort[relation.to_.name], 'syseq_r2')
            conjs.append(z3.Or(
                # Relaxed
                # z3.And(
                #     # z3.Implies(self.associated_in_relation(rname, rstate1, r1, r2), self.associated_in_relation(rname, rstate2, r1, r2)),
                #     # z3.Implies(self.associated_in_relation(rname, rstate2, r1, r2), self.associated_in_relation(rname, rstate1, r1, r2))
                # ),
                # Strict
                sys1.relation_states[rname] == sys2.relation_states[rname],
            ))

        return z3.And(conjs)

    def associated_in_relation(self, rname: str, rstate: z3.ExprRef, ref1: z3.ExprRef, ref2: z3.ExprRef) -> z3.ExprRef:
        raise NotImplementedError()
        relation = self.relations[rname]
        from_name = relation.from_.name
        to_name = relation.to_.name
        from_ref_sort = self.ref_sort[from_name]
        to_ref_sort = self.ref_sort[to_name]
        optional_from_ref = self.get_or_define_optional(from_ref_sort)
        optional_to_ref = self.get_or_define_optional(to_ref_sort)
        if relation.kind == RelationKind.ONE_ONE:
            forward = self.oneone[rname].getForward
            return optional_to_ref.just(ref2) == z3.Select(forward(rstate), ref1)
        elif relation.kind == RelationKind.MANY_ONE:
            forward = self.manyone[rname].getForward
            return optional_to_ref.just(ref2) == z3.Select(forward(rstate), ref1)
        else:
            forward = self.manymany[rname].getForward
            return z3.IsMember(ref2, z3.Select(forward(rstate), ref1))


    def generate_model_state(self, model: str, prefix=''):
        ret = self.fresh(self.set_sort[model], prefix + 'set_' + model)
        # self.solver.add(self.well_formed_model_state(model, ret))
        return ret

    def generate_relation_state(self, rname: str, prefix=''):
        r = self.relations[rname]
        if r.kind == RelationKind.ONE_ONE:
            ret = self.fresh(self.oneone[rname], prefix + "r11" + rname)
        elif r.kind == RelationKind.MANY_ONE:
            ret = self.fresh(self.manyone[rname], prefix + "r13" + rname)
        elif r.kind == RelationKind.MANY_MANY:
            ret = self.fresh(self.manymany[rname], prefix + "r33" + rname)
        else:
            raise RuntimeError("Unknown relation kind %s" % str(r.kind))
        # self.solver.add(self.well_formed_relation_state(rname, ret))
        return ret

    def obj_to_ref(self, model: str, obj: z3.ExprRef) -> z3.ExprRef:
        pfields = self.models[model].primary_fields()
        vals = [self.obj_projs[model][p.name](obj) for p in pfields]
        return self.ref_mk[model](*vals)

    def generate_system_state(self, prefix=''):
        ret = SystemState(self.generate_model_states(prefix), self.generate_relation_states(prefix))
        self.add(self.well_formed_system_state(ret))
        return ret

    def generate_model_states(self, prefix=''):
        return {model_name: self.generate_model_state(model_name, prefix) for model_name in self.models.keys()}

    def generate_relation_states(self, prefix=''):
        return {rel_name: self.generate_relation_state(rel_name, prefix) for rel_name in self.relations.keys()}

    def assert_no_unique(self, shadow, sys: 'SystemState'):
        context = self
        for (argname, mname) in shadow.unique_ids.items():
            mstate = sys.models[mname]
            context.solver.add(z3.Not(z3.IsMember(
                ToRef(mname, W(shadow.args[argname])).eval(context, sys),
                context.set_ids[mname](mstate))))

        # unique_ids also not present in relation states
        for (rname, r) in self.relations.items():
            if r.from_.name == mname:
                rstate = sys.relations[rname]
                if r.kind == RelationKind.ONE_ONE:
                    forward = context.oneone[rname].getForward(rstate)
                    optional_from_sort = context.get_or_define_optional(context.ref_sort[r.from_.name])
                    optional_to_sort = context.get_or_define_optional(context.ref_sort[r.to_.name])
                    context.solver.add(optional_to_sort.is_nothing(z3.Select(forward, ToRef(mname, W(shadow.args[argname])).eval(context, sys))))
                elif r.kind == RelationKind.MANY_ONE:
                    forward = context.manyone[rname].getForward(rstate)
                    optional_to_sort = context.get_or_define_optional(context.ref_sort[r.to_.name])
                    context.solver.add(optional_to_sort.is_nothing(z3.Select(forward, ToRef(mname, W(shadow.args[argname])).eval(context, sys))))
                else:
                    to_ref = context.ref_sort[r.to_.name]
                    forward = context.manyone[rname].getForward(rstate)
                    forward_id = z3.Select(forward, ToRef(mname, W(shadow.args[argname])).eval(context, sys))
                    context.solver.add(z3.EmptySet(to_ref) == forward_id)


class AST:
    def cached_pprint(self, depth) -> str:
        if hasattr(self, '_cached_pp'):
            return self._cached_pp
        else:
            self._cached_pp = self.pprint(depth)
            return self._cached_pp

    def pprint(self, depth) -> str:
        """Pretty-print the AST."""
        raise NotImplementedError(f'pprint catch-all class {type(self)}')

    def __repr__(self) -> str:
        return self.cached_pprint(0)


class Expr(AST):
    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        """Prefer checked_eval() unless you have good reasons."""
        raise NotImplementedError('eval catch-all')

    def checked_eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        """If z3 raises an exception, this method will print a message
        for debugging.

        """
        import traceback
        try:
            res = self.eval(context, sys)
            return res
        except z3.z3types.Z3Exception:
            logging.error('z3 failed due to expr: {}'.format(self.cached_pprint(0)))
            raise
        except KeyError:
            logging.error('KeyError during {}'.format(self.cached_pprint(0)))
            traceback.print_exc()
            raise
        except:
            logging.error('other error due to expr: {}'.format(self.cached_pprint(0)))
            traceback.print_exc()
            raise

    def type(self, context: Context) -> Type:
        raise NotImplementedError('type catch-all')


class W(Expr):
    """Directly wrap a z3 expr as an expression. Useful for testing and caching results (see SetF and SaveObj)."""
    def __init__(self, thing: z3.ExprRef, type: Optional[Type] = None):
        self.thing = thing
        self._type = type

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        return self.thing

    def pprint(self, depth):
        return str(self.thing)

    def type(self, context: Context) -> Optional[Type]:
        if self.type:
            return self._type
        else:
            raise RuntimeError('W does not have an associated type')


class Free(Expr):
    """Named constant.

    A Free expression always evaluates to a fresh constant that is
    shared in this path.  The name always refers to an "arg", and the
    Z3 constant is generated (freshly) in Op.generate_args().

    """
    def __init__(self, name: str, sort, typ: Type):
        self.name = name
        self.sort = sort
        self.typ = typ

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        # return z3.Const(self.name, self.sort)
        return context.get_free_val(self.name)

    def pprint(self, depth) -> str:
        return self.name

    def type(self, context):
        if self.typ:
            return self.typ
        else:
            raise NotImplementedError('W does not have an associated type')

class FreeObj(Expr):
    """See also Free.  Same as Free, the name is only shared within a Path."""
    def __init__(self, model: str, name: str):
        self.model = model
        self.name = name

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        # return z3.Const(self.name, context.obj_sort[self.model])
        return context.get_free_val(self.name)

    def pprint(self, depth) -> str:
        return self.name

    def type(self, context) -> Type:
        return Type.OBJ(self.model)

class Literal(Expr):
    def __init__(self, value):
        self.value = value

    def pprint(self, depth) -> str:
        return repr(self.value)


class Str(Literal):
    def __init__(self, value: str):
        if value is None:
            value = ''
        super().__init__(value)

    def eval(self, context, sys) -> z3.ExprRef:
        return z3.StringVal(self.value)

    def sort(self, context):
        return z3.StringSort()

    def type(self, context):
        return Type.STR()


class Int(Literal):
    def __init__(self, value: int):
        assert isinstance(value, int)
        super().__init__(value)

    def eval(self, context, sys) -> z3.ExprRef:
        return z3.IntVal(self.value)

    def sort(self, context):
        return z3.IntSort()

    def type(self, context):
        return Type.INT()


class Bool(Literal):
    def __init__(self, value: int):
        super().__init__(value)

    def eval(self, context, sys) -> z3.ExprRef:
        return z3.BoolVal(self.value)

    def sort(self, context):
        return z3.BoolSort()

    def type(self, context):
        return Type.BOOL()


class Real(Literal):
    def __init__(self, value: float):
        super().__init__(value)

    def eval(self, context, sys) -> z3.ExprRef:
        return z3.RealVal(self.value)

    def sort(self, context):
        return z3.RealSort()

    def type(self, context):
        return Type.REAL()


class Seq(Literal):
    def __init__(self, value: list):
        super().__init__(value)

    def eval(self, context, sys) -> z3.ExprRef:
        raise NotImplementedError()

    def sort(self, context):
        raise NotImplementedError()

    def type(self, context):
        raise NotImplementedError('Seq type not supported')

class Any(Expr):
    fn = 'any'

    def __init__(self, model: str, qs: Expr):
        assert isinstance(qs, Expr)
        self.model = model
        self.qs = qs

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        qs = self.qs.checked_eval(context, sys)
        ids = context.set_ids[self.model](qs)
        data = context.set_data[self.model](qs)
        ref = z3.FreshConst(context.ref_sort[self.model], 'any_ref')
        context.add(z3.IsMember(ref, ids))
        return z3.Select(data, ref)

    def pprint(self, depth) -> str:
        return f'any[{self.model}]({self.qs.cached_pprint(depth)})'

    def type(self, context) -> Type:
        return Type.OBJ(self.model)


class GetF(Expr):
    fn = 'getf'

    def __init__(self, model: str, field: str, obj: Expr):
        assert isinstance(obj, Expr)
        assert isinstance(model, str)
        assert isinstance(field, str)
        self.model = model
        self.field = field
        self.obj = obj

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        obj = self.obj.checked_eval(context, sys)
        assert obj is not None
        return context.obj_projs[self.model][self.field](obj)

    def pprint(self, depth) -> str:
        return f'getf[{self.model}]({self.field}, {self.obj.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        m = context.models[self.model]
        return m.fields[self.field].ty



class SetFs(Expr):
    fn = 'setfs'

    def __init__(self, model: str, pairs: Dict[str, Expr], obj: Expr):
        self.model = model
        self.pairs = pairs
        self.obj = obj

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        model = context.models[self.model]
        fields = model.data_fields()
        obj = self.obj.checked_eval(context, sys)
        vals = [GetF(self.model, f.name, W(obj)).checked_eval(context, sys) for f in fields]
        for field, value in self.pairs.items():
            idx = model.field_to_idx[field]
            vals[idx] = value.checked_eval(context, sys)
        newobj = context.obj_mk[self.model](*vals)
        return newobj

    def type(self, context: Context) -> Type:
        return Type.OBJ(self.model)

    def pprint(self, depth):
        return f'setfs[{self.model}]({".".join(f"{k}={v.cached_pprint(0)}" for k, v in self.pairs.items())})'


class SetF(SetFs):
    def __init__(self, model: str, field: str, value: Expr, obj: Expr):
        assert isinstance(model, str)
        assert isinstance(field, str)
        assert isinstance(value, Expr)
        assert isinstance(obj, Expr)
        self.model = model
        self.field = field
        self.value = value
        self.obj = obj
        super().__init__(model, {field: value}, obj)

    def pprint(self, depth) -> str:
        return f'setf[{self.model}]({self.field}, {self.value.pprint(depth)}, {self.obj.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        return Type.OBJ(self.model)


class Unary(Expr):
    def __init__(self, op, inner):
        self.op = op
        self.inner = inner
        assert isinstance(self.inner, Expr)

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        a = self.inner.checked_eval(context, sys)
        if   self.op == 'not': return z3.Not(a)
        elif self.op == 'toint': return z3.ToInt(a)
        elif self.op == 'tofloat': return z3.ToFloat(a)
        elif self.op == 'tostr': raise NotImplementedError('ToStr not implemented yet')
        else:
            raise ValueError(f'Unknown unary operator: {self.op}')

    def pprint(self, depth) -> str:
        return f'{self.op}({self.inner.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        if self.op == 'not':
            return Type.BOOL()
        elif self.op == 'toint':
            return Type.INT()
        elif self.op == 'tofloat':
            return Type.REAL()
        elif self.op == 'tostr':
            return Type.STR()
        else:
            raise ValueError('Unknown unary op: {}'.format(self.op))


class Binary(Expr):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        # print('Binary.eval', self.left.pprint(0), self.left.type(context), self.right.pprint(0), self.right.type(context))
        a = self.left.checked_eval(context, sys)
        b = self.right.checked_eval(context, sys)
        try:
            if   self.op == '+': return a + b
            elif self.op == '-': return a - b
            elif self.op == '*': return a * b
            elif self.op == '/': return a / b   # z3 does not support //
            elif self.op == '<': return a < b
            elif self.op == '>': return a > b
            elif self.op == '<=': return a <= b
            elif self.op == '>=': return a >= b
            elif self.op == '==': return (a == b)
            elif self.op == '!=': return z3.Not(a == b)
            elif self.op == '&&': return z3.And(a, b)
            elif self.op == '||': return z3.Or(a, b)
            elif self.op == '^': return z3.Xor(a, b)
            else:
                raise ValueError(f'Unknown binary operator: {self.op}')
        except:
            logging.error('exc during binary, a={a} ({atype} {asort}), b={b} ({btype} {bsort})'.format(
                a=self.left.cached_pprint(0), atype=self.left.type(context), asort=a.sort(),
                b=self.right.cached_pprint(0), btype=self.right.type(context), bsort=b.sort()
            ))
            raise

    def pprint(self, depth) -> str:
        return f'({self.left.cached_pprint(depth)}){self.op}({self.right.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        arith_ops = {'+', '-', '*', '/'}
        comp_ops = {'>', '<', '>=', '<=', '==', '!='}
        bool_ops = {'&&', '||', '^'}
        if self.op in arith_ops:
            ty = self.left.type(context)
            return ty
        elif self.op in comp_ops or self.op in bool_ops:
            return Type.BOOL()
        else:
            raise ValueError('Unknown binary op: {}'.format(self.op))


class ToRef(Expr):
    """Wrap raw data into a refernce."""
    fn = 'toref'

    def __init__(self, model: str, *values):
        self.model = model
        self.values = values

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        values = [x.checked_eval(context, sys) for x in self.values]
        return context.ref_mk[self.model](*values)

    def pprint(self, depth) -> str:
        # return f'⟦{",".join([x.cached_pprint(depth) for x in self.values])}⟧'
        return f'toref[{self.model}]({",".join([x.cached_pprint(depth) for x in self.values])})'

    def type(self, context: Context) -> Type:
        return Type.REF(self.model)


class ObjToRef(Expr):
    """Convert an object to a reference."""
    fn = 'objtoref'

    def __init__(self, model: str, obj: Expr):
        assert isinstance(obj, Expr)
        self.model = model
        self.obj = obj

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        if isinstance(self.obj, Deref):
            # Short-cut the Deref case because ObjToRef(Deref(ref)) == ref.
            return self.obj.ref.checked_eval(context, sys)
        else:
            obj = self.obj.checked_eval(context, sys)
            return context.obj_to_ref(self.model, obj)

    def pprint(self, depth) -> str:
        if isinstance(self.obj, Deref):
            return self.obj.ref.pprint(depth)
        else:
            return f'{self.obj.cached_pprint(depth)}.id'

    def type(self, context: Context) -> Type:
        assert self.obj.type(context) == Type.OBJ(self.model)
        return Type.REF(self.model)

class All(Expr):
    fn = 'all'

    def __init__(self, model: str):
        self.model = model

    def eval(self, context, sys: SystemState) -> z3.ExprRef:
        return sys.models[self.model]

    def pprint(self, depth) -> str:
        return f'all[{self.model}]'

    def type(self, context) -> Type:
        return Type.SET(self.model)


class Exists(Expr):
    """
    Check if an object exists in the current system state.
    """
    fn = 'exists'

    def __init__(self, model: str, ref: Expr):
        assert isinstance(model, str)
        assert isinstance(ref, Expr)
        self.model = model
        self.ref = ref

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        ids = context.set_ids[self.model](sys.models[self.model])
        # HACK
        if self.ref.type(context) in [Type.INT(), Type.STR()]:
            self.ref = ToRef(self.model, self.ref)
        ref = self.ref.checked_eval(context, sys)
        # HACK
        if ref.sort() in [z3.IntSort(), z3.StringSort()]:
            ref = context.ref_mk[self.model](ref)
        return z3.IsMember(ref, ids)

    def pprint(self, depth) -> str:
        assert isinstance(self.ref, Expr)
        return f'exists[{self.model}]({self.ref.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        return Type.BOOL()


class Empty(Expr):
    """
    Returns True if the queryset is empty.
    """
    fn = 'empty'

    def __init__(self, model: str, qs: Expr) -> None:
        assert isinstance(qs, Expr)
        self.model = model
        self.qs = qs

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        qs = self.qs.checked_eval(context, sys)
        ids = context.set_ids[self.model](qs)
        ref_sort = context.ref_sort[self.model]
        return ids != z3.EmptySet(ref_sort)

    def pprint(self, depth) -> str:
        return f'empty[{self.model}]({self.qs.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        return Type.BOOL()


class Filteq(Expr):
    """
    Filteq is a restricted form of Filter.

    Filteq(field, value, qs) is semantically equivalent to Filter(λobj -> obj.field==value, qs).
    """
    fn = 'filteq'

    def __init__(self, model: str, field: str, rhs: Expr, qs: Expr, comparator: str = 'eq', reverse: bool = False):
        assert isinstance(rhs, Expr)
        assert isinstance(qs, Expr)
        self.model = model
        self.field = field
        self.rhs = rhs
        self.qs = qs
        self.comparator = comparator
        self.reverse = reverse

    def eval(self, context: Context, sys: SystemState):
        qs = self.qs.checked_eval(context, sys)
        rhs = self.rhs.checked_eval(context, sys)
        data = context.set_data[self.model](qs)
        ids = context.set_ids[self.model](qs)
        order = context.set_order[self.model](qs)
        newdata = data
        neworder = order
        x = z3.FreshConst(context.ref_sort[self.model], 'filteq_ref_')
        lhs = GetF(self.model, self.field, W(z3.Select(data, x))).checked_eval(context, sys)
        # obj in filter(cond, qs) iff obj in qs && cond(obj)
        selection_formula = None
        if self.comparator == 'eq':
            selection_formula = (lhs == rhs)
        elif self.comparator == 'gt':
            selection_formula = (lhs > rhs)
        elif self.comparator == 'lt':
            selection_formula = (lhs < rhs)
        else:
            raise ValueError('filteq only supports eq, gt, lt, not: ' + self.comparator)
        if self.reverse:
            selection_formula = z3.Not(selection_formula)
        newids = z3.Lambda([x], z3.If(z3.And([z3.IsMember(x, ids), selection_formula]),
                                      z3.BoolVal(True),
                                      z3.BoolVal(False)))
        return context.set_mk[self.model](newdata, newids, neworder)

    def pprint(self, depth) -> str:
        if not self.reverse:
            fn = 'filteq'
        else:
            fn = 'exclude'
        return f'{fn}[{self.model}]({self.field}=={self.rhs.cached_pprint(depth)},{self.qs.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        return Type.SET(self.model)

class FilteqRel(Expr):
    """
    FilteqRel(rel, field, rhs, qs) corresponds to qs.filter(rel_field=rhs).
    """
    fn = 'filteqrel'

    def __init__(self, relation: str, field: str, rhs: Expr, qs: Expr, reverse: bool = False):
        assert isinstance(rhs, Expr)
        assert isinstance(qs, Expr)
        self.relation = relation
        self.field = field
        self.rhs = rhs
        self.qs = qs
        assert reverse == False, 'Not implemented'

    def eval(self, context: Context, sys: SystemState):
        # Let R = filteqrel(rel, field, rhs, qs).
        # R is a subset of qs.
        # x in R iff x is related to some y in state[rel.to], and y.field = rhs.

        # Let T be the subset of state[rel.to] such that for all y in T, y.field = rhs.
        r = context.relations[self.relation]
        to_name = r.to_.name
        T = Filteq(to_name,
                   self.field, self.rhs,
                   W(sys.model_states[to_name],
                     Type.SET(to_name))).checked_eval(context, sys)

        # Find R' such that for all x in R', x is related to some object in T.
        R_ = Follow(self.relation, 'backward', W(T)).checked_eval(context, sys)

        # Restrict R' to be a subset of qs.
        from_name = r.from_.name
        qs = self.qs.checked_eval(context, sys)
        qs_data = context.set_data[from_name](qs)
        qs_ids = context.set_ids[from_name](qs)
        qs_order = context.set_order[from_name](qs)
        R_ids = context.set_ids[from_name](R_)
        res_data = qs_data   # qs_data is probably simpler
        res_ids = z3.SetIntersect(R_ids, qs_ids)
        res_order = qs_order
        res = context.set_mk[from_name](res_data, res_ids, res_order)
        return res

    def pprint(self, depth) -> str:
        return f'filteqrel({self.relation},{self.field}=={self.rhs.cached_pprint(depth)},{self.qs.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        return Type.SET(context.relations[self.relation].from_.name)


class GFilteqRel(Expr):
    """
    Generalized version of FilteqRel.  Once it's tested, both `FilteqRel` and `Filteq` should be removed in favor of this.

    gfilteqrel([rel1, rel2, rel3], f, val, qs) == qs.filter(rel1__rel2__rel3__f=val)
    """
    fn = 'gfilteqrel'

    def __init__(self, relations: List[str], field: str, rhs: Expr, qs: Expr):
        assert isinstance(relations, list)
        assert isinstance(field, str)
        assert isinstance(rhs, Expr)
        assert isinstance(qs, Expr)
        assert len(relations) > 0     # Use `GetF` in that case.
        self.relations = relations
        self.field = field
        self.rhs = rhs
        self.qs = qs

    def type(self, context: Context) -> Type:
        # check self.qs with rels[0]
        rels = [context.relations[rname] for rname in self.relations]
        qs_ty = self.qs.type(context)
        assert rels[0].from_.name == qs_ty.model

        # check relation chain
        cur_model = rels[0].from_.name
        for rel in rels:
            assert rel.from_.name == cur_model
            cur_model = rel.to_.name
        cur_model = context.models[cur_model]

        # check field_ty with self.rhs
        expected_field_ty = cur_model.fields[self.field].ty
        supplied_field_ty = self.rhs.type(context)
        assert expected_field_ty == supplied_field_ty

        return qs_ty

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        rels = [context.relations[rname] for rname in self.relations]

        # Starting from the final model.
        # Restrict the final model state such that its field
        # `self.field` is specified.
        cur_model_state = sys.models[rels[-1].to_.name]
        cur_model_state_data = context.set_data[rels[-1].to_.name](cur_model_state)
        cur_model_state_ids  =  context.set_ids[rels[-1].to_.name](cur_model_state)
        cur_model_state_order=context.set_order[rels[-1].to_.name](cur_model_state)
        x = z3.FreshConst(context.ref_sort[rels[-1].to_.name], 'gfilteqrel_ref_')
        cur_model_state_ids = z3.Lambda([x], z3.If(z3.And([z3.IsMember(x, cur_model_state_ids), (z3.Select(cur_model_state_data, x) == self.rhs)])),
                                        z3.BoolVal(True), z3.BoolVal(False))
        cur_model_state = context.set_mk[rels[-1].to_.name](cur_model_state_data, cur_model_state_ids, cur_model_state_order)

        for rel in reversed(rels):
            rname = rel.name
            cur_model_state = Follow(rname, 'backward', cur_model_state).checked_eval(context, sys)

        # First model reached. Intersect it with the input qs to find
        # the result.
        first_data = context.set_data[rels[0].from_](cur_model_state)
        first_ids  =  context.set_ids[rels[0].from_](cur_model_state)
        first_order=context.set_order[rels[0].from_](cur_model_state)
        res_ids = z3.SetIntersect(first_ids, self.qs)
        res_data = first_data
        res_order = first_order
        res_model_state = context.set_mk[rels[0].from_](res_data, res_ids, res_order)
        return res_model_state


class Union(Expr):
    fn = 'union'

    def __init__(self, model: str, qs1: Expr, qs2: Expr):
        self.model = model
        self.qs1 = qs1
        self.qs2 = qs2

    def eval(self, context: Context, sys: SystemState):
        qs1 = self.qs1.checked_eval(context, sys)
        qs2 = self.qs2.checked_eval(context, sys)
        return MergeModelSet(context, self.model, qs2, qs1)

    def type(self, context: Context) -> Type:
        assert self.qs1.type(context) == self.qs2.type(context)
        return Type.SET(self.model)


class Filter(Expr):
    fn = 'filter'

    def __init__(self, model: str, cond: Expr, qs: Expr):
        # NOTE(function): cond for now should evaluate to a z3 function of sort: Object -> Bool.
        # e.g.: W(z3.Lambda([x], ...))
        self.model = model
        self.cond = cond
        self.qs = qs

    def eval(self, context: Context, sys: SystemState):
        cond = self.cond.checked_eval(context, sys)
        qs = self.qs.checked_eval(context, sys)
        data = context.set_data[self.model](qs)
        ids = context.set_ids[self.model](qs)
        order = context.set_order[self.model](qs)
        newdata = data
        neworder = order
        x = z3.FreshConst(context.ref_sort[self.model], 'filter_ref_')
        # obj in filter(cond, qs) iff obj in qs && cond(obj)
        newids = z3.Lambda([x], z3.If(z3.And([z3.IsMember(x, ids), cond(z3.Select(data, x))])),
                                      z3.BoolVal(True),
                                      z3.BoolVal(False))
        return context.set_mk[self.model](newdata, newids, neworder)

    def type(self, context: Context) -> Type:
        return Type.SET(self.model)


class Map(Expr):
    fn = 'map'

    def __init__(self, model, func, qs):
        self.model = model
        self.func = func
        self.qs = qs

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        func = self.func.checked_eval(context, sys)
        qs = self.qs.checked_eval(context, sys)
        data = context.set_data[self.model](qs)
        ids = context.set_ids[self.model](qs)
        order = context.set_order[self.model](qs)
        x = z3.FreshConst(context.ref_sort[self.model], 'map_ref_')
        newdata = z3.Lambda([x], func(z3.Select(data, x)))
        newids = ids
        neworder = order
        return context.set_mk[self.model](newdata, newids, neworder)

    def type(self, context: Context) -> Type:
        raise NotImplementedError('Map.type(): Map not implemented')


class Deref(Expr):
    fn = 'deref'

    def __init__(self, model: str, qs: Expr, ref: Expr):
        assert isinstance(model, str)
        assert isinstance(qs, Expr)
        assert isinstance(ref, Expr)
        self.model = model
        self.qs = qs
        self.ref = ref

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        if self.ref.type(context) in [Type.STR(), Type.INT()]:
            ref = ToRef(self.model, self.ref)
        else:
            ref = self.ref
        ref = ref.checked_eval(context, sys)
        # Try to extract something here.
        obj = z3.Select(context.set_data[self.model](self.qs.checked_eval(context, sys)),
                        ref)
        # However, if id does not exist in qs, the object can carry a
        # non-sensical ID (since it is not covered by WF1). Forbid it.
        context.solver.add(ObjToRef(self.model, W(obj)).checked_eval(context, sys) == ref)
        return obj

    def pprint(self, depth) -> str:
        if isinstance(self.qs, All):
            return f'deref[{self.model}]({self.ref.cached_pprint(depth)})'
        else:
            return f'deref[{self.model}]({self.qs.cached_pprint(depth)},{self.ref.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        return Type.OBJ(self.model)


class Count(Expr):
    fn = 'count'

    def __init__(self, model, qs):
        self.model = model
        self.qs = qs

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        # NOTE(context): added axioms for fresh variables
        k = context.fresh(z3.IntSort(), 'count_int')
        data = context.set_data[self.model](sys.models[self.model])
        # FIXME: unfortunately, z3 has dropped the support for set cardinality
        # context.add(z3.SetHasSize(data, k))
        context.add(k >= 0)
        return k

    def pprint(self, depth) -> str:
        return f'count({self.qs.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        return Type.INT()


class Follow(Expr):
    fn = 'follow'

    def __init__(self, relation: str, direction: str, qs: Expr):
        assert isinstance(qs, Expr)
        assert isinstance(relation, str)
        assert isinstance(direction, str)
        self.relation = relation
        self.direction = direction
        self.qs = qs

    def pprint(self, depth) -> str:
        return f'follow[{self.direction}]({self.relation},{self.qs.cached_pprint(0)})'

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        r = context.relations[self.relation]
        if r.kind == RelationKind.ONE_ONE:
            return self.follow_oneone(context, sys, r)
        elif r.kind == RelationKind.MANY_ONE:
            return self.follow_manyone(context, sys, r)
        elif r.kind == RelationKind.MANY_MANY:
            return self.follow_manymany(context, sys, r)
        else:
            raise ValueError('Unknown relation kind %s' % str(r.kind))

    def follow_oneone(self, context: Context, sys: SystemState, r: Relation):
        qs = self.qs.checked_eval(context, sys)
        from_sort = context.ref_sort[r.from_.name]
        to_sort = context.ref_sort[r.to_.name]
        optional_from_sort = context.get_or_define_optional(from_sort)
        optional_to_sort = context.get_or_define_optional(to_sort)
        rs = sys.relations[r.name]
        if self.direction == 'forward':
            ids = context.set_ids[r.from_.name](qs)
            y = z3.FreshConst(to_sort, 'follow11_to_ref_')
            backward = context.oneone[r.name].getBackward(rs)
            ret_data = context.set_data[r.to_.name](sys.models[r.to_.name])
            ret_ids = z3.Lambda([y], z3.If(
                optional_from_sort.is_just(z3.Select(backward, y)),
                z3.IsMember(optional_from_sort.fromJust(z3.Select(backward, y)), ids),
                False
            ))
            ret_order = context.set_order[r.to_.name](sys.models[r.to_.name])
            return context.set_mk[r.to_.name](ret_data, ret_ids, ret_order)
        elif self.direction == 'backward':
            ids = context.set_ids[r.to_.name](qs)
            x = z3.FreshConst(from_sort, 'follow11_from_ref_')
            forward = context.oneone[r.name].getForward(rs)
            ret_data = context.set_data[r.from_.name](sys.models[r.from_.name])
            ret_ids = z3.Lambda([x], z3.If(
                optional_to_sort.is_just(z3.Select(forward, x)),
                z3.IsMember(optional_to_sort.fromJust(z3.Select(forward, x)), ids),
                False
            ))
            ret_order = context.set_order[r.from_.name](sys.models[r.from_.name])
            return context.set_mk[r.from_.name](ret_data, ret_ids, ret_order)
        else:
            raise ValueError('Unknown follow direction %s' % self.direction)

    def follow_manyone(self, context: Context, sys: SystemState, r: Relation):
        qs = self.qs.checked_eval(context, sys)
        from_sort = context.ref_sort[r.from_.name]
        to_sort = context.ref_sort[r.to_.name]
        optional_to_sort = context.get_or_define_optional(to_sort)
        rs = sys.relations[r.name]
        if self.direction == 'forward':
            # y is in the result set, if and only if exists x in qs, such that just y = rs.forward[x]
            # which is equal to rs.backward[y] /\ qs != empty
            from_ids = context.set_ids[r.from_.name](qs)
            backward = context.manyone[r.name].getBackward(rs)
            y = z3.FreshConst(to_sort, 'follow31_to_ref_')
            ret_data = context.set_data[r.to_.name](sys.models[r.to_.name])
            ret_ids = z3.Lambda([y], z3.SetIntersect(z3.Select(backward, y), from_ids) != z3.EmptySet(from_sort))
            ret_order = context.set_order[r.to_.name](sys.models[r.to_.name])
            return context.set_mk[r.to_.name](ret_data, ret_ids, ret_order)
        elif self.direction == 'backward':
            to_ids = context.set_ids[r.to_.name](qs)
            forward = context.manyone[r.name].getForward(rs)
            x = z3.FreshConst(from_sort, 'follow31_from_ref_')
            ret_data = context.set_data[r.from_.name](sys.models[r.from_.name])
            ret_ids = z3.Lambda([x], z3.If(
                optional_to_sort.is_just(z3.Select(forward, x)),
                z3.IsMember(optional_to_sort.fromJust(z3.Select(forward, x)), to_ids),
                False
            ))
            ret_order = context.set_order[r.from_.name](sys.models[r.from_.name])
            return context.set_mk[r.from_.name](ret_data, ret_ids, ret_order)
        else:
            raise ValueError('Unknown follow direction %s' % self.direction)

    def follow_manymany(self, context: Context, sys: SystemState, r: Relation):
        qs = self.qs.checked_eval(context, sys)
        from_sort = context.ref_sort[r.from_.name]
        to_sort = context.ref_sort[r.to_.name]
        rs = sys.relations[r.name]
        if self.direction == 'forward':
            from_ids = context.set_ids[r.from_.name](qs)
            backward = context.manymany[r.name].getBackward(rs)
            y = z3.FreshConst(to_sort, 'follow33_to_ref_')
            ret_data = context.set_data[r.to_.name](sys.models[r.to_.name])
            ret_ids = z3.Lambda([y], z3.SetIntersect(z3.Select(backward, y), from_ids) != z3.EmptySet(from_sort))
            ret_order = context.set_order[r.to_.name](sys.models[r.to_.name])
            return context.set_mk[r.to_.name](ret_data, ret_ids, ret_order)
        elif self.direction == 'backward':
            to_ids = context.set_ids[r.to_.name](qs)
            forward = context.manymany[r.name].getForward(rs)
            x = z3.FreshConst(from_sort, 'follow33_from_from_')
            ret_data = context.set_data[r.from_.name](sys.models[r.from_.name])
            ret_ids = z3.Lambda([x], z3.SetIntersect(z3.Select(forward, x), to_ids) != z3.EmptySet(to_sort))
            ret_order = context.set_order[r.from_.name](sys.models[r.from_.name])
            return context.set_mk[r.from_.name](ret_data, ret_ids, ret_order)
        else:
            raise ValueError('Unknown follow direction %s' % self.direction)

    def type(self, context: Context) -> Type:
        relation = context.relations[self.relation]
        if self.direction == 'forward':
            return Type.SET(relation.to_.name)
        elif self.direction == 'backward':
            return Type.SET(relation.from_.name)
        else:
            raise ValueError('Unknown follow direction %s' % self.direction)



class Limit(Expr):
    fn = 'limit'

    def __init__(self, model, n, qs):
        self.model = model
        self.n = n
        self.qs = qs

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        # NOTE(context): added axioms for fresh vars
        res = context.fresh(context.set_sort[self.model], 'limit_set')
        res_data = context.set_data[self.model](res)
        res_ids = context.set_ids[self.model](res)
        res_ids_cnt = Count(self.model, res_ids).checked_eval(context, sys)
        context.add(res_ids_cnt <= self.n)
        context.add((res_data == context.set_data[self.model](sys.models[self.model])))
        return res

    def pprint(self, depth) -> str:
        return f'limit[{self.model}]({self.n}, {self.qs.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        return Type.SET(self.model)



class First(Expr):
    fn = 'first'

    def __init__(self, model, qs):
        self.model = model
        self.qs = qs

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        qs = self.qs.checked_eval(context, sys)
        ids = context.set_ids[self.model](qs)
        data = context.set_data[self.model](qs)
        order = context.set_order[self.model](qs)
        ret_obj = z3.FreshConst(context.obj_sort[self.model], 'first_obj_')
        ret_ref = ObjToRef(W(obj)).check_eval(context, sys)
        ret_order = z3.Select(order, ref)
        x_ref = z3.FreshConst(context.ref_sort[self.model], 'first_ref_')
        x_order = z3.Select(order, x_ref)
        # NOTE(context): the returned object is the object with the
        # smallest 'order number'.
        context.add(z3.ForAll([x_ref], z3.And([
            z3.IsMember(ret_ref, ids),    # the returned object is valid.
            z3.Implies(
                z3.IsMember(x_ref, ids),  # the returned object has the smallest
                ret_order <= x_order      # order number.
            )
        ])))
        return obj

    def pprint(self, depth) -> str:
        return f'first[{self.model}]({self.qs.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        return Type.OBJ(self.model)


class Last(Expr):
    fn = 'last'

    def __init__(self, model, qs):
        self.model = model
        self.qs = qs

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        qs = self.qs.checked_eval(context, sys)
        ids = context.set_ids[self.model](qs)
        data = context.set_data[self.model](qs)
        order = context.set_order[self.model](qs)
        ret_obj = z3.FreshConst(context.obj_sort[self.model], 'last_obj_')
        ret_ref = ObjToRef(W(obj)).check_eval(context, sys)
        ret_order = z3.Select(order, ref)
        x_ref = z3.FreshConst(context.ref_sort[self.model], 'last_ref_')
        x_order = z3.Select(order, x_ref)
        # NOTE(context): the returned object is the object with the
        # smallest 'order number'.
        context.add(z3.ForAll([x_ref], z3.And([
            z3.IsMember(ret_ref, ids),    # the returned object is valid.
            z3.Implies(
                z3.IsMember(x_ref, ids),  # the returned object has the largest
                ret_order >= x_order      # order number.
            )
        ])))
        return obj

    def pprint(self, depth) -> str:
        return f'last[{self.model}]({self.qs.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        return Type.OBJ(self.model)


class OrderBy(Expr):
    fn = 'orderby'

    def __init__(self, model: str, field: str, order: str, qs: Expr):
        assert isinstance(qs, Expr)
        self.model = model
        self.field = field
        self.order = order
        self.qs = qs

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        qs = self.qs.checked_eval(context, sys)

        # Find out the type of the field.
        model = context.models[self.model]
        field = model.fields[self.field]

        # Get the value of the field.
        data = context.set_data[self.model](qs)
        ids = context.set_ids[self.model](qs)
        ref = z3.FreshConst(context.ref_sort[self.model], 'orderby_ref_')
        obj_of_x = z3.Select(data, ref)
        field_of_obj = GetF(self.model, self.field, W(obj_of_x)).checked_eval(context, sys)

        # If the field type is integer, we just replace its .
        # The correctness is tricky, depending on other expressions.
        if field.ty == Type.INT():
            if self.order == 'asc':
                order = z3.Lambda([ref], field_of_obj)
            elif self.order == 'desc':
                order = z3.Lambda([ref], z3.IntVal(0) - field_of_obj)
            else:
                raise ValueError('')
        else:
            # Be conservative if we don't support this.
            order = z3.FreshConst(context.models[self.model].order_sort(), 'orderby_order_')

        return context.set_mk[self.model](data, ids, order)

    def pprint(self, depth) -> str:
        return f'orderby[{self.model}, {self.field}, {self.order}]({self.qs.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        return Type.SET(self.model)


class Reverse(Expr):
    """Reverse the objects in a queryset."""

    fn = 'reverse'

    def __init__(self, model: str, qs: Expr):
        assert isinstance(qs, Expr)
        self.model = model
        self.qs = qs

    def eval(context, sys):
        qs = self.qs.checked_eval(context, sys)
        data = context.set_data[self.model](qs)
        ids = context.set_ids[self.model](qs)
        order = context.set_order[self.model](order)
        x = z3.FreshConst(context.ref_sort[self.model], 'reverse_ref_')
        neworder = z3.Lambda([x], z3.IntVal(0) - z3.Select(order, x))
        return context.set_mk[self.model](data, ids, neworder)


class Aggregate(Expr):
    fn = 'aggregate'

    def __init__(self, model: str, field: str, kind: str, qs: Expr):
        self.model = model
        self.field = field
        self.kind = kind
        self.qs = qs

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        qs = self.qs.checked_eval(context, sys)
        field_ty = context.models[self.model].fields[self.field].ty
        field_sort = field_ty.smt(context)
        ref_sort = context.ref_sort[self.model]
        getIds = context.set_ids[self.model]
        getData = context.set_data[self.model]
        res = context.fresh(field_sort, 'aggregate_res')
        ref = context.fresh(ref_sort, 'aggregate_ref')
        if self.kind == 'max':
            # res is bigger than any valid field
            context.add(z3.ForAll(
                [ref],
                z3.Implies(z3.IsMember(getIds(qs), ref),
                           res >= GetF(self.model, self.field, W(z3.Select(getData(qs), ref))).checked_eval(context, sys))
            ))
        elif self.kind == 'min':
            # res is smaller than any valid field
            context.add(z3.ForAll(
                [ref],
                z3.Implies(z3.IsMember(getIds(qs), ref),
                           res <= GetF(self.model, self.field, W(z3.Select(getData(qs), ref))).checked_eval(context, sys))
            ))
        elif self.kind == 'avg':
            # TODO
            logging.warn('avg not yet supported in aggregate')
        elif self.kind == 'sum':
            # TODO
            logging.warn('sum not yet supported in aggregate')
        else:
            raise ValueError('aggregate kind %s not supported' % self.kind)
        return res

    def type(self, context: Context) -> Type:
        raise NotImplementedError('Aggregate.type() TODO')


class Singleton(Expr):
    fn = 'singleton'

    def __init__(self, model: str, obj: Expr):
        self.model = model
        self.obj = obj

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        ref_sort = context.ref_sort[self.model]
        obj = self.obj.checked_eval(context, sys)  # eval'd
        pvals = [GetF(self.model, pf.name, W(obj)).checked_eval(context, sys) for pf in context.models[self.model].primary_fields()]
        ref = context.ref_mk[self.model](*pvals)
        ids = z3.SetAdd(z3.EmptySet(ref_sort), ref)
        data = z3.K(ref_sort, obj)
        # NOTE: give this object an unknown order.
        order = z3.FreshConst(context.models[self.model].order_sort(), 'singleton_order')
        return context.set_mk[self.model](data, ids, order)

    def pprint(self, depth) -> str:
        return f'singleton[{self.model}]({self.obj.cached_pprint(depth)})'

    def type(self, context: Context) -> Type:
        return Type.SET(self.model)


class RegexMatch(Expr):
    '''Regex match.'''
    fn = 'match'

    def __init__(self, pattern: str, str: Expr):
        self.pattern = pattern
        self.str = str

    def eval(self, context: Context, sys: SystemState) -> z3.ExprRef:
        pattern = z3.Re(self.pattern)
        str = self.str.checked_eval(context, sys)
        return z3.InRe(str, pattern)

    def pprint(self, depth) -> str:
        return f'match({self.pattern}, {self.str.cached_pprint(depth)})'

    def type(self, _context: Context) -> Type:
        return Type.BOOL()


class Command(AST):
    def apply(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        raise NotImplementedError('command apply catch-all')


class Delete(Command):
    def __init__(self, model: str, qs: Expr):
        self.model = model
        self.qs = qs

    def apply(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        newsys = deepcopy(sys)
        newsys.models[self.model] = ExcludeModelSet(context, self.model, sys.models[self.model], self.qs.checked_eval(context, sys))
        return newsys, []

    def pprint(self, depth) -> str:
        return f'delete[{self.model}]({self.qs.cached_pprint(depth)})'


class Update(Command):
    def __init__(self, model: str, qs: Expr):
        self.model = model
        self.qs = qs

    def apply(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        newsys = deepcopy(sys)
        newsys.models[self.model] = MergeModelSet(context, self.model, sys.models[self.model], self.qs.checked_eval(context, sys))
        return newsys, []

    def pprint(self, depth) -> str:
        return f'update[{self.model}]({self.qs.cached_pprint(depth)})'


class SaveObj(Command):
    def __init__(self, model: str, obj: Expr):
        self.model = model
        self.obj = obj

    def apply(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        obj = self.obj.checked_eval(context, sys)
        pvals = [GetF(self.model, pf.name, W(obj)).checked_eval(context, sys) for pf in context.models[self.model].primary_fields()]
        ref = context.ref_mk[self.model](*pvals)
        qs = sys.models[self.model]
        newsys = deepcopy(sys)
        olddata = context.set_data[self.model](qs)
        oldids = context.set_ids[self.model](qs)
        oldorder = context.set_order[self.model](qs)
        x = z3.FreshConst(context.ref_sort[self.model], 'saveobj_ref_')
        newsys.models[self.model] = context.set_mk[self.model](
            z3.Store(olddata, ref, obj),
            z3.SetAdd(oldids, ref),
            # give this object a fresh order if this object is new.
            z3.Lambda([x], z3.If(
                z3.IsMember(x, oldrefs),
                z3.Select(oldorder, x),
                z3.FreshConst(z3.IntSort(), 'saveobj_int_')))
        )
        return newsys, []


class Guard(Command):
    def __init__(self, cond: Expr):
        self.cond = cond

    def apply(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        return sys, [self.cond.checked_eval(context, sys)]


class Link(Command):
    def __init__(self, relation: str, ref1: Expr, ref2: Expr):
        assert isinstance(ref1, Expr)
        assert isinstance(ref2, Expr)
        self.relation = relation
        self.ref1 = ref1
        self.ref2 = ref2

    def pprint(self, depth) -> str:
        return f'link[{self.relation}]({self.ref1.cached_pprint(depth)}, {self.ref2.cached_pprint(depth)})'

    def apply(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        kind = context.relations[self.relation].kind
        if kind == RelationKind.ONE_ONE:
            return self.link_oneone(context, sys)
        elif kind == RelationKind.MANY_ONE:
            return self.link_manyone(context, sys)
        else:
            return self.link_manymany(context, sys)

    def link_oneone(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        from_ref_sort = context.ref_sort[context.relations[self.relation].from_.name]
        to_ref_sort = context.ref_sort[context.relations[self.relation].to_.name]
        optional_from_sort = context.get_or_define_optional(from_ref_sort)
        optional_to_sort = context.get_or_define_optional(to_ref_sort)
        ref1 = self.ref1.checked_eval(context, sys)
        ref2 = self.ref2.checked_eval(context, sys)
        rs = sys.relations[self.relation]
        forward = context.oneone[self.relation].getForward(rs)
        backward = context.oneone[self.relation].getBackward(rs)
        newforward = z3.Store(forward, ref1, optional_to_sort.just(ref2))
        newbackward = z3.Store(backward, ref2, optional_from_sort.just(ref1))
        newrs = context.oneone[self.relation].create(newforward, newbackward)
        newsys = deepcopy(sys)
        newsys.relations[self.relation] = newrs
        return newsys, []

    def link_manyone(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        to_ref_sort = context.ref_sort[context.relations[self.relation].to_.name]
        optional_to_sort = context.get_or_define_optional(to_ref_sort)
        ref1 = self.ref1.checked_eval(context, sys)
        ref2 = self.ref2.checked_eval(context, sys)
        rs = sys.relations[self.relation]
        forward = context.manyone[self.relation].getForward(rs)
        backward = context.manyone[self.relation].getBackward(rs)
        newforward = z3.Store(forward, ref1, optional_to_sort.just(ref2))
        newbackward = z3.Store(backward, ref2, z3.SetAdd(z3.Select(backward, ref2), ref1))
        assert(forward.sort() == newforward.sort())
        assert(backward.sort() == newbackward.sort())
        newrs = context.manyone[self.relation].create(newforward, newbackward)
        newsys = deepcopy(sys)
        newsys.relations[self.relation] = newrs
        return newsys, []

    def link_manymany(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        ref1 = self.ref1.checked_eval(context, sys)
        ref2 = self.ref2.checked_eval(context, sys)
        rs = sys.relations[self.relation]
        forward = context.manymany[self.relation].getForward(rs)
        backward = context.manymany[self.relation].getBackward(rs)
        newforward = z3.SetAdd(z3.Select(forward, ref1), ref2)
        newbackward = z3.SetAdd(z3.Select(backward, ref2), ref1)
        newrs = context.manymany[self.relation].create(newforward, newbackward)
        newsys = deepcopy(sys)
        newsys.relations[self.relation] = newrs
        return newsys, []


class RLink(Command):
    """
    Handle cases like y.related_set = xs
    """
    def __init__(self, relation: str, y: Expr, xs: Expr):
        self.relation = relation
        self.y = y
        self.xs = xs

    def pprint(self, depth: int) -> str:
        return f'rlink[{self.relation}]({self.y.cached_pprint(depth)}, {self.xs.cached_pprint(depth)})'

    def apply(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        relation = context.relations[self.relation]
        if relation.kind == RelationKind.MANY_ONE:
            return self.rlink_manyone(context, sys)
        elif relation.kind == RelationKind.MANY_MANY:
            return self.rlink_manymany(context, sys)
        else:
            return self.rlink_oneone(context, sys)

    def rlink_manyone(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        # Given
        #   class A:
        #     r = ForeignKey(B)
        #   class B: ...
        # A_B_r is a many-one relation from A to B.
        # Suppose xs is A set, y is B, then
        #   y.r_set = xs
        # should link every obj in xs to y, that is:
        #   A_B_r'.f = \z -> if z in xs then just(y) else A_B_r.f[z]
        #   A_B_r'.b = \z -> if z == y  then xs      else A_B_r.b[z]
        r = context.relations[self.relation]
        rs = sys.relations[self.relation]
        forward, backward = context.destruct_relation_state(r, rs)
        optional_to_ref = context.get_or_define_optional(context.ref_sort[r.to_.name])
        y = self.y.checked_eval(context, sys)
        xs = self.xs.checked_eval(context, sys)
        z = z3.FreshConst(r.from_.ref_sort(), 'rlink31_ref1_')
        newforward = z3.Lambda([z], z3.If(z3.IsMember(z, xs), optional_to_ref.just(y), forward[z]))
        z = z3.FreshConst(r.to_.ref_sort(), 'rlink31_ref2_')
        newbackward = z3.Lambda([z], z3.If(z == y, xs, backward[z]))
        newrs = context.manyone[self.relation](newforward, newbackward)
        newsys = deepcopy(sys)
        newsys.relations[self.relation] = newrs
        return newsys, []

    def rlink_manymany(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        # Given
        #   class A:
        #     r = ManyToManyField(B)
        #   class B: ...
        # A_B_r is a many-many relation from A to B.
        # Suppose xs is A set, y is B, then
        #   y.r_set = xs
        # should be interpreted as:
        #   r'.f = \z -> if z in xs then r.f[z] + y else r.f[z]
        #   r'.b = \z -> if z == y  then xs         else r.b[z]
        y = self.y.checked_eval(context, sys)
        xs = self.xs.checked_eval(context, sys)
        r = context.relations[self.relation]
        rs = sys.relations[self.relation]
        forward, backward = context.destruct_relation_state(r, rs)
        z = z3.FreshConst(r.from_.ref_sort(), 'rlink33_ref1_')
        newforward = z3.Lambda([z], z3.IsMember(z, xs), z3.SetAdd(forward[z], y), forward[z])
        z = z3.FreshConst(r.from_.ref_sort(), 'rlink33_ref2_')
        newbackward = z3.Lambda([z], z3.If(z == y, xs, backward[z]))
        newrs = context.manymany[self.relation](newforward, newbackward)
        newsys = deepcopy(sys)
        newsys.relations[self.relation] = newrs
        return newsys, []

    def rlink_oneone(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        # Given
        #  class A:
        #    r = OneToOneField(B)
        #  class B: ...
        # A_B_r is a one-one relation from A to B.
        # Suppose x is A, y is B, then
        #   y.r_set = x
        # should be interpreted as:
        #   A_B_r.f[x] = just(y)
        #   A_B_r.b[y] = just(x)
        r = context.relations[self.relation]
        rs = sys.relations[self.relation]
        forward, backward = context.destruct_relation_state(r, rs)
        optional_from_ref = context.get_or_define_optional(context.ref_sort[r.from_.name])
        optional_to_ref = context.get_or_define_optional(context.ref_sort[r.to_.name])
        x = self.xs.checked_eval(context, sys)
        y = self.y.checked_eval(context, sys)
        newforward = z3.Store(forward, x, optional_to_ref.just(y))
        newbackward = z3.Store(backward, y, optional_from_ref.just(x))
        newrs = context.oneone[self.relation].create(newforward, newbackward)
        newsys = deepcopy(sys)
        newsys.relations[self.relation] = newrs
        return newsys, []


class LinkObj(Command):
    """
    A specialized variant of `Link` which works on objects instead of references.

    Semantically, LinkObj(o1, o2) is equivalent to Link(r1, r2) where o1.id = r1, o2.id = r2.

    This could and should be rewritten using `ObjToRef`, but this is more convenient.
    """
    def __init__(self, relation: str, obj1: Expr, obj2: Expr):
        self.relation = relation
        self.obj1 = obj1
        self.obj2 = obj2

    def pprint(self, depth) -> str:
        return f'linkobj[{self.relation}]({self.obj1.cached_pprint(depth)}, {self.obj2.cached_pprint(depth)})'

    def apply(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        r = context.relations[self.relation]
        obj1 = self.obj1.checked_eval(context, sys)
        obj2 = self.obj2.checked_eval(context, sys)
        # print('LinkObj:', r.name, r.kind, r.from_, r.to_, self.obj1.type(context).model, self.obj2.type(context).model)
        assert self.obj1.type(context).model == r.from_.name
        assert self.obj2.type(context).model == r.to_.name
        ref1 = context.obj_to_ref(r.from_.name, obj1)
        ref2 = context.obj_to_ref(r.to_.name, obj2)
        link_cmd = Link(self.relation, W(ref1, Type.REF(r.from_.name)), W(ref2, Type.REF(r.to_.name)))
        return link_cmd.apply(context, sys)


class Unlink(Command):
    def __init__(self, relation: str, ref1: Expr, ref2: Expr):
        self.relation = relation
        self.ref1 = ref1
        self.ref2 = ref2

    def apply(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        relation = context.relations[self.relation]
        if relation.kind == RelationKind.ONE_ONE:
            return self.unlink_oneone(context, sys)
        elif relation.kind == RelationKind.MANY_ONE:
            return self.unlink_manyone(context, sys)
        else:
            return self.unlink_manymany(context, sys)

    def unlink_oneone(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        optional_from_sort = context.get_or_define_optional(context.ref_sort[context.relations[self.relation].from_.name])
        optional_to_sort = context.get_or_define_optional(context.ref_sort[context.relations[self.relation].to_.name])
        ref1 = self.ref1.checked_eval(context, sys)
        ref2 = self.ref2.checked_eval(context, sys)
        rs = sys.relations[self.relation]
        forward = context.manymany[self.relation].getForward(rs)
        backward = context.manymany[self.relation].getBackward(rs)
        newforward = z3.Store(forward, ref1, optional_to_sort.nothing)
        newbackward = z3.Store(backward, ref2, optional_from_sort.nothing)
        newrs = context.manymany[self.relation].create(newforward, newbackward)
        newsys = deepcopy(sys)
        newsys.relations[self.relation] = newrs
        return newsys, []

    def unlink_manyone(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        optional_to_sort = context.get_or_define_optional(context.ref_sort[context.relations[self.relation].to_.name])
        ref1 = self.ref1.checked_eval(context, sys)
        ref2 = self.ref2.checked_eval(context, sys)
        rs = sys.relations[self.relation]
        forward = context.manymany[self.relation].getForward(rs)
        backward = context.manymany[self.relation].getBackward(rs)
        newforward = z3.Store(forward, ref1, optional_to_sort.nothing)
        newbackward = z3.SetDel(z3.Select(backward, ref2), ref1)
        newrs = context.manyone[self.relation].create(newforward, newbackward)
        newsys = deepcopy(sys)
        newsys.relations[self.relation] = newrs
        return newsys, []

    def unlink_manymany(self, context: Context, sys: SystemState) -> Tuple[SystemState, List[z3.ExprRef]]:
        ref1 = self.ref1.checked_eval(context, sys)
        ref2 = self.ref2.checked_eval(context, sys)
        rs = sys.relations[self.relation]
        forward = context.manymany[self.relation].getForward(rs)
        backward = context.manymany[self.relation].getBackward(rs)
        newforward = z3.SetDel(z3.Select(forward, ref1), ref2)
        newbackward = z3.SetDel(z3.Select(backward, ref2), ref1)
        newrs = context.manyone[self.relation].create(newforward, newbackward)
        newsys = deepcopy(sys)
        newsys.relations[self.relation] = newrs
        return newsys, []


class PairChecker:
    """PairChecker checks a pair of operations.

    @hook is called before each solver.check with arguments ('com' |
    'sem') and the solver.

    """

    def __init__(self,
                 models: Optional[List[Model]],
                 relations: Optional[List[Relation]],
                 timeout: float,
                 hook: Optional[Callable[[str, z3.Solver], None]] = None,
                 ):
        self.models = OrderedDict({model.name: model for model in models}) if models else OrderedDict()
        self.relations = OrderedDict({rel.name: rel for rel in relations}) if relations else OrderedDict()
        self.context = Context(z3.Solver(), self.models, self.relations)
        self.hook = hook

        # We do not use z3's timeout mechanism, unless _dbg_sync is true.
        self.timeout = timeout
        self._dbg_sync = False

        # Enable parallelism
        z3.set_param('parallel.enable', True)

    def commutativity_rule(self, P: Op, Q: Op):
        S = self.context.generate_system_state('S_')
        Ps = P.generate_shadow(self.context, 'P_')
        Qs = Q.generate_shadow(self.context, 'Q_')

        Porig = self.context.generate_system_state('Porig_')
        Qorig = self.context.generate_system_state('Qorig_')
        Pok = Ps.judge(self.context, Porig)
        Qok = Qs.judge(self.context, Qorig)

        SPQ = Qs.apply(self.context, Ps.apply(self.context, S, prefix='SP_'), prefix='SPQ_')
        self.context.add(self.context.well_formed_system_state(SPQ))

        SQP = Ps.apply(self.context, Qs.apply(self.context, S, prefix='SQ_'), prefix='SQP_')
        self.context.add(self.context.well_formed_system_state(SPQ))

        unique_vars = Ps.smt_unique_vars() + Qs.smt_unique_vars()
        distinct_ids = z3.Distinct(unique_vars) if len(unique_vars) > 1 else z3.BoolVal(True)
        vars = Ps.smt_vars() + Qs.smt_vars() + S.smt_vars() + Porig.smt_vars() + Qorig.smt_vars()
        self.context.add(z3.And([Pok, Qok, distinct_ids]))
        return z3.Not(self.context.sys_eq(SPQ, SQP))

    def _check_commutativity(self, P: Op, Q: Op):
        rule = self.commutativity_rule(P, Q)
        try:
            self.context.solver.push()
            # print('rule: ', rule)
            self.context.add(rule)
            if self.hook:
                self.hook('com', self.context.solver)
            ret = self.context.solver.check()
            # Debugging code:
            # if ret == z3.sat:
            #     m = self.context.solver.model()
            #     for decl in m:
            #         # print(decl, ' : ', type(m[decl]))
            #         if isinstance(m[decl], z3.FuncInterp):
            #             print(decl, '->', m[decl])
            #         else:
            #             print(decl, '->', simplify(m[decl]))
        finally:
            self.context.solver.pop()
        return ret

    def check_commutativity(self, P: Op, Q: Op):
        if self.timeout is not None:
            if self._dbg_sync:
                self.context.solver.set('timeout', int(self.timeout * 1000))
                return self._check_commutativity(P, Q)
            else:
                return timedcall(self.timeout, self._check_commutativity, P, Q)
        else:
            return self._check_commutativity(P, Q)

    def precondition_rule_single(self, P: Op, Q: Op, prefix=''):
        S = self.context.generate_system_state(prefix + '_S_')
        Ps = P.generate_shadow(self.context, prefix + '_P_')
        Qs = Q.generate_shadow(self.context, prefix + '_Q_')

        # Q is a valid shadow
        Qorig = self.context.generate_system_state(prefix+'Qorig')
        self.context.add(self.context.well_formed_system_state(Qorig))
        Qok = Qs.judge(self.context, Qorig)
        self.context.add(Qok)

        # Initially, P is valid on S.
        onlyP_ok = Ps.judge(self.context, S)
        self.context.add(self.context.well_formed_system_state(S))
        self.context.add(onlyP_ok)

        # P is still valid AFTER Q.
        SQ = Qs.apply(self.context, S, prefix=prefix + '_inserted_')
        self.context.add(self.context.well_formed_system_state(SQ))
        QthenP_ok = Ps.judge(self.context, SQ)

        # globally unique vars
        unique_vars = Ps.smt_unique_vars() + Qs.smt_unique_vars()
        if len(unique_vars) > 1:
            distinct_ids = z3.Distinct(unique_vars)
            self.context.add(distinct_ids)

        return z3.Not(QthenP_ok)

    def _check_precondition(self, P: Op, Q: Op):
        # ok(P) => ok(Q->P)
        try:
            self.context.solver.push()
            rule = self.precondition_rule_single(P, Q, 'sem1')
            self.context.add(rule)
            if self.hook:
                self.hook('sem1', self.context.solver)
            ret = self.context.solver.check()
        finally:
            self.context.solver.pop()
        if ret == z3.sat:
            return z3.sat
        # ok(Q) => ok(P->Q)
        try:
            self.context.solver.push()
            rule = self.precondition_rule_single(P, Q, 'sem2')
            self.context.add(rule)
            if self.hook:
                self.hook('sem2', self.context.solver)
            ret = self.context.solver.check()
        finally:
            self.context.solver.pop()
        return ret

    def check_precondition(self, P: Op, Q: Op):
        if self.timeout is not None:
            if self._dbg_sync:
                self.context.solver.set('timeout', int(self.timeout * 1000))
                return self._check_precondition(P, Q)
            else:
                return timedcall(self.timeout, self._check_precondition, P, Q)
        else:
            return self._check_precondition(P, Q)

    ###########################################################
    # Check if two code paths are independent from each other #
    ###########################################################
    def independent_rule(self, P: Op, Q: Op):
        S = self.context.generate_system_state('S_')
        Ps = P.generate_shadow(self.context, 'P_')
        Qs = Q.generate_shadow(self.context, 'Q_')
        Porig = self.context.generate_system_state('Porig_')
        Qorig = self.context.generate_system_state('Qorig_')
        Pok = Ps.judge(self.context, Porig)
        Qok = Qs.judge(self.context, Qorig)
        P_indep_Q = z3.Implies(
            Ps.judge(self.context, Qs.apply(self.context, S)),
            Ps.judge(self.context, S),
        )
        Q_indep_P = z3.Implies(
            Qs.judge(self.context, Ps.apply(self.context, S)),
            Qs.judge(self.context, S),
        )
        unique_vars = Ps.smt_unique_vars() + Qs.smt_unique_vars()
        distinct_ids = z3.Distinct(unique_vars) if len(unique_vars) > 1 else z3.BoolVal(True)
        vars = Ps.smt_vars() + Qs.smt_vars() + S.smt_vars() + Porig.smt_vars() + Qorig.smt_vars()
        return z3.Not(z3.ForAll(vars, z3.Implies(
             z3.And([self.context.well_formed_system_state(S),
                     self.context.well_formed_system_state(Porig),
                     self.context.well_formed_system_state(Qorig),
                     Pok, Qok, distinct_ids]),
            z3.And([P_indep_Q, Q_indep_P])
        )))

    def check_independent(self, P: Op, Q: Op):
        if self.timeout is not None:
            if self._dbg_sync:
                self.context.solver.set('timeout', int(self.timeout * 1000))
                return self._check_independent(P, Q)
            else:
                return timedcall(self.timeout, self._check_independent, P, Q)
        else:
            return self._check_independent(P, Q)

    def _check_independent(self, P: Op, Q: Op):
        rule = self.independent_rule(P, Q)
        try:
            self.context.solver.push()
            self.context.add(rule)
            if self.hook:
                self.hook('indep', self.context.solver)
            ret = self.context.solver.check()
        finally:
            self.context.solver.pop()
        return ret

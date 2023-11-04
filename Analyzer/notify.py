from Analyzer.path import Path
from typing import Union

from Verifier.checker import Type, Expr, FreeObj, Free


_current_path: Union[Path, None] = None


def set_current_path(path):
    global _current_path
    assert isinstance(path, Path)
    _current_path = path


def get_current_path() -> Path:
    global _current_path
    assert isinstance(_current_path, Path)
    return _current_path


def is_during_analysis() -> bool:
    return _current_path is not None


def add_effect(effect):
    path = get_current_path()
    path.add_effect(effect)


def add_cond(cond):
    path = get_current_path()
    path.add_path_cond(cond)


def add_arg(arg: str, type=None):
    assert isinstance(arg, str)
    path = get_current_path()
    path.add_free_var(arg, type)


def add_unique_id(id_name, mname: str):
    assert isinstance(id_name, str)
    global _current_path
    assert _current_path is not None
    return _current_path.add_unique_id(id_name, mname)


def new_insert_obj_name():
    global _current_path
    assert _current_path is not None
    return _current_path.fresh_insert_obj_name()


def obtain_free_expr(name: str, ty: Type) -> Free:
    assert isinstance(ty, Type)
    expr = Free(name, ty.smt(None), ty)
    path = get_current_path()
    path.add_free_var(name, ty)
    return expr


def obtain_free_obj_expr(model: str) -> FreeObj:
    obj_name = new_insert_obj_name()
    path = get_current_path()
    obj_expr = FreeObj(model, obj_name)
    path.add_free_var(obj_name, Type.OBJ(model))
    return obj_expr

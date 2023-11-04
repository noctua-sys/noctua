from typing import Optional, Tuple
from Verifier.checker import Model, Relation, RelationKind, Type
from django.db import models
from django.contrib.postgres.fields import JSONField
from django.db.models.fields.files import ImageField

def simple_field_to_type(f: models.Field) -> Optional[Type]:
    """
    Find the IR type of a *simple* Django field object.
    """
    string_kind = [models.BinaryField, models.CharField, models.EmailField, models.FileField, models.FilePathField, models.ImageField, models.GenericIPAddressField, models.SlugField, models.TextField, models.URLField, models.UUIDField, JSONField, ImageField]
    boolean_kind = [models.BooleanField]
    integer_kind = [models.AutoField, models.BigAutoField, models.BigIntegerField, models.IntegerField, models.PositiveIntegerField, models.PositiveSmallIntegerField, models.SmallIntegerField] # models.PositiveBigIntegerField, models.SmallAutoField
    real_kind = [models.DecimalField, models.DurationField, models.FloatField]
    datetime_kind = [models.DateField, models.DateTimeField, models.TimeField]
    type_map = {
        Type.STR(): string_kind,
        Type.BOOL(): boolean_kind,
        Type.INT(): integer_kind,
        Type.REAL(): real_kind,
        Type.DATE(): datetime_kind
    }
    for ty, acceptlist in type_map.items():
        for cls in acceptlist:
            if isinstance(f, cls):
                return ty
    return None


def related_key_to_relation(django_model, key) -> Tuple[str, str]:
    """Returns (relation name, direction).

    obj.key == Follow(name, dir, set_of_this_django_model)."""
    descriptor = getattr(django_model, key)
    fld = descriptor.field
    model_name = django_model._meta.label
    from_label = fld.model._meta.label
    to_label = fld.related_model._meta.label
    relation_name = f'{from_label}__{to_label}__{fld.name}'
    if from_label == model_name:
        dir = 'forward'
    else:
        dir = 'backward'
    return relation_name, dir


def related_key_to_related_model(django_model, key) -> Tuple[type, bool]:
    """Returns (the django model for obj.key, object?)"""
    from django.db.models.fields.related import ForeignKey, OneToOneField
    descriptor = getattr(django_model, key)
    fld = descriptor.field
    this_model_label = django_model._meta.label
    from_label = fld.model._meta.label

    if isinstance(fld, OneToOneField):
        obj = True
        if this_model_label == from_label:
            that_model = fld.related_model
        else:
            that_model = fld.model
    elif isinstance(fld, ForeignKey):
        if this_model_label == from_label:
            obj = True
            that_model = fld.related_model
        else:
            obj = False
            that_model = fld.model
    else:
        obj = False
        if this_model_label == from_label:
            that_model = fld.related_model
        else:
            that_model = fld.model

    return that_model, obj

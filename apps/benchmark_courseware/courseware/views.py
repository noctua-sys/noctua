from django.http import HttpResponse
from courseware.models import *
from django.views.decorators.http import require_http_methods
from Verifier.checker import Type


def extract_params(*argspecs):
    """@argspecs is a list of pairs of argument names and argument types.

    The argument type should be bool, int, float, or str.  Any other
    types are not supported.

    """
    # Sanity check: argument type must be supported.
    for (_, argtyp) in argspecs:
        assert argtyp in [bool, int, float, str]

    # Decorator:
    # - if not during analysis, extract argument types from request.POST
    # - if during analysis, declare arguments
    def decorator(view_fn):
        def wrapper_view_fn(req, *args, **kwargs):
            from Analyzer import notify
            extracted = []
            if notify.is_during_analysis():
                for (argname, argtyp) in argspecs:
                    argval = req.POST._getitem_with_pytype(argname, argtyp)
                    extracted.append(argval)
            else:
                for (argname, argtyp) in argspecs:
                    argval = req.POST[argname]
                    argval = argtyp(argval)
                    extracted.append(argval)
            return view_fn(req, *tuple(extracted + list(args)), **kwargs)
        return wrapper_view_fn
    return decorator


@require_http_methods(['POST'])
def Register(request):
    student = Student.objects.create()
    return HttpResponse('success')


@require_http_methods(['POST'])
def AddCourse(request):
    course = Course.objects.create()
    return HttpResponse('success')


@require_http_methods(['POST'])
@extract_params(('sid', int), ('cid', int))
def Enroll(request, sid, cid):
    student = Student.objects.get(pk=sid)
    course  = Course.objects.get(pk=cid)
    Enrolment.objects.create(student=student,course=course)
    return HttpResponse('success')


@require_http_methods(['POST'])
@extract_params(('cid', int))
def DeleteCourse(request, cid):
    Course.objects.filter(pk=cid).delete()
    return HttpResponse('success')

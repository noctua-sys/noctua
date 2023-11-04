from django.http import HttpResponse
from smallbank.models import *
from django.views.decorators.http import require_http_methods
from Verifier.checker import Type


def extract_params(argspecs):
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
@extract_params([('cust1_id', int), ('total', int)])
def Amalgamate(req, cust1_id, total):
    """
    Move cust1's checking to cust1's savings.
    """
    # To match AutoGR's code
    cust1 = Account.objects.get(pk=cust1_id)
    assert 0 < total <= cust1.checking_bal
    cust1.savings_bal += total
    cust1.checking_bal -= total
    cust1.save()
    return HttpResponse('')
    # cust1 = Account.objects.get(pk=cust1_id)
    # cust2 = Account.objects.get(pk=cust2_id)
    # cust2.savings_bal += cust1.savings_bal
    # cust1.checking_bal = 0
    # cust1.save()
    # cust2.save()
    # return HttpResponse('')

@require_http_methods(['GET'])
@extract_params([('cust_name', str)])
def Balance(req, cust_name):
    """Return the total balance."""
    cust = Account.objects.get(name=cust_name)
    balance = cust.savings_bal + cust.checking_bal
    return HttpResponse(str(balance))


@require_http_methods(['POST'])
@extract_params([('cust_name', str), ('amount', float)])
def DepositChecking(req, cust_name, amount):
    """Deposit into the checking balance."""
    assert amount > 0
    cust = Account.objects.get(name=cust_name)
    cust.checking_bal += amount
    cust.save()
    return HttpResponse('')


@require_http_methods(['POST'])
@extract_params([('cust1_id', int), ('cust2_id', int), ('amount', float)])
def SendPayment(req, cust1_id, cust2_id, amount):
    """Send from cust1's checking balance to cust2's checking balance."""
    assert amount > 0
    cust1 = Account.objects.get(pk=cust1_id)
    cust2 = Account.objects.get(pk=cust2_id)
    if amount > cust1.checking_bal:
        raise RuntimeError('Insufficient funds')
    cust1.checking_bal -= amount
    cust2.checking_bal += amount
    cust1.save()
    cust2.save()
    return HttpResponse('')


@require_http_methods(['POST'])
@extract_params([('cust_name', str), ('amount', float)])
def TransactSavings(req, cust_name, amount):
    assert amount > 0
    cust = Account.objects.get(name=cust_name)
    if amount > cust.savings_bal:
        raise RuntimeError('Insufficient funds')
    cust.savings_bal -= amount
    cust.save()
    return HttpResponse('')


@require_http_methods(['POST'])
@extract_params([('cust_name', str), ('amount', float)])
def WriteCheck(req, cust_name, amount):
    assert amount > 0
    cust = Account.objects.get(name=cust_name)
    total = cust.savings_bal + cust.checking_bal
    if amount <= total:
        cust.checking_bal -= amount
    else:
        raise RuntimeError('Insufficient funds')
    cust.save()
    return HttpResponse('')

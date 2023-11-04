from Analyzer.symbolic import Sym

def print_sym(prompt, obj):
    print(prompt, 'Python:', type(obj), str(obj))
    if isinstance(obj, Sym):
        print(prompt, 'Sym type:', obj.type)
        print(prompt, 'Sym expr:', obj.expr)

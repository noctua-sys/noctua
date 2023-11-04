def is_being_analyzed():
    from Analyzer.notify import _current_path
    return _current_path is not None

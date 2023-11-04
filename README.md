# soir

First, follow this document to set up a virtual environment.

- To run consistency analysis: ./manage.py consistency
- To run E2E benchmarks: see e2e/readme.md

Python 3.9 is preferred, but in case you don't have it, follow [Notes on Python 3.10].

## How to use Analyzer

Add `Analyzer` to the list of INSTALLED_APPS, and you'll get a command called `consistency` in `manage.py`.

Options:

* **`--urlconf`**
* **`--timeout`**: default: `1` (1 second).  How much time can be spent on one check at most.
* **`--dir`**: default: `.`.  The directory where all the output files go.
* **`--suffix`**: default: empty string. This string will be appended to every output filename. Usually used to avoid filename collisions when multiple instances are running in parallel.
* **`--analyze-regexp`**: default: `.*`.  Only IDs matching this regexp will be analyzed and verified.
* **`--verify-regexp`**: default: `.*`.  If ID1 and ID2 match this regexp, the pair (ID1,ID2) will be checked.
* **`--save-regexp`**: optional, and unset by default.  If ID matches this regexp, the check will be recorded in a file (in the SMTLib2 format) specified by SAVE-FILENAME-FORMAT.
* **`--save-filename-format`**: default `{kind}-{id1}-{id2}-{suffix}.smt2`.
* **`--independence`**: default `False`.  Enable the independence check.

Example:

```bash
./manage.py consistency --timeout=5 --dir=foo --suffix=5 --save-regexp='.*'
```

## Notes on development

1. Create a virtualenv

```bash
mkdir -p ~/.virtualenvs
python3 -m venv ~/.virtualenvs/analyzer
```

2. Activate it

```bash
source ~/.virtualenvs/analyzer/bin/activate
```

3. Install dependencies

```bash
pip3 install -r requirements.txt
```

This installs the offical versions of dependencies into your virtualenv. You must Python search paths to make Python aware of our modified versions of these dependencies.

4. Configure Python search paths

For PyRight, set `extraPaths`. See also [PyRight's doc](https://github.com/microsoft/pyright/blob/main/docs/configuration.md).

For virtualenv, modify `~/.virtualenvs/analyzer/bin/activate`, add the following line:

```python
export PYTHONPATH=/PATH/TO/deps/django22:/PATH/TO/deps/restframework
```

5. Verify your development environment is ready.

```bash
$ source ~/.virtualenvs/analyzer/bin/activate
$ python3
>>> import rest_framework
You are using a customized version of Django 2.2.28.dev20220301132326
You are using a customized version of djangorestframework 3.13.1
```

## Notes on the use of Nix

Just run `nix-shell` under the directory of an application. You will need to re-run `nix-shell` whenever you modify code.

## Notes on Python 3.10

Python 3.10 made some incompatible changes to `collections`. Therefore, some code changes are required:

```
pip3 install pytz --upgrade
```

```python
# site-packages/corsheaders/checks.py
# site-packages/jwt/api_jwt.py
# site-packages/jwt/api_jws.py
# and so on

from collections import xxx     # old
from collections.abc import xxx # new
```

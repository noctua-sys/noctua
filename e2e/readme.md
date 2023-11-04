# test coordinator

```bash
python3 -m Coord.server --config=e2e/test_spec.json
```

```python
import Coord
from Coord.client import coord
id = coord.add(1, 'foo')  # return 1
coord.add(1, 'foo')       # return False
coord.remove(1)           # return True
```

# run E2E benchmarks

**YOU MUST USE THE ORIGINAL DJANGO.**

That is, you have to use the following `PYTHONPATH`:

```
export PYTHONPATH=$NOCTUA
```

The specs for zhihu and postgraduation have been written:

1. zhihu.json
2. pg.json

Run `runpg.sh` and `runzhihu.sh` to start a local experiment.

You may want to adjust `EXP_NAME` and `SC_FLAG` defined in these scripts.

```
./runpg.sh ${WRITE_RATIO} ${EXP_SUFFIX}
# for example
./runpg.sh 0.15 1
```

The results can be found in this folder.


from enum import Enum, auto


class FpMode(Enum):
    # Default.  Use Z3's RealSort.  This mode is usually faster.
    FAST = auto()

    # Use Z3's FPSort.  Slower but the semantics are exact.
    EXACT = auto()


# TODO: Currently we only support the FAST mode.
FP_MODE = FpMode.FAST

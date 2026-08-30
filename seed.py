import os
import random
import numpy as np


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHEDSEED"] = str(seed)

set_determinism(42)
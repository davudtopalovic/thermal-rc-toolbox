import math
def get_scale(x):
    if x == 0:
        return 0.0
    exponent = int(math.floor(math.log10(abs(x))))
    return float(10 ** exponent)
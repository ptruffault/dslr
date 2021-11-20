import math
import numpy as np


def is_numeric(i):
    return (isinstance(i, int) or isinstance(i, float)) and not math.isnan(i)



def mymean(d):
    if d is None:
        return 0
    l = len(d)
    if l == 0:
        return 0;
    return mysum(d) / l;


def mylen(d):
    if d is None:
        return 0
    i = 0
    for l in d:
        i += 1
    return i


def mymin(d):
    if d is None:
        return None
    ret = None
    for line in d:
        if is_numeric(line) and ret is None or ret < line:
            ret = line
    return ret


def mymax(d):
    if d is None:
        return None
    ret = None
    for line in d:
        if is_numeric(line) and ret is None or ret > line:
            ret = line
    return ret


def mysum(d):
    if d is None:
        return 0
    ret = 0
    for line in d:
        if is_numeric(line):
            ret += line
    return ret


def mystd(d):
    if d is None:
        return 0
    ret = 0
    mean = mymean(d)
    for l in d:
        if is_numeric(l):
            ret = ret + (l - mean) ** 2
    ret = math.sqrt((ret / (mylen(d) - 1)))
    return ret


def percent(d, per):
    if d is None:
        return 0
    i = (per / 100) * (mylen(d))
    if i.is_integer():
        return d[int(i) - 1]
    else:
        i_up = math.ceil(i) - 1
        return d[i_up]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

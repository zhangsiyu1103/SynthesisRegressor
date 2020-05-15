import os
import torch
import numpy as np
import math


#def polynomial(*args):
#    ret_str = ""
#    def poly(x, *args):
#        ret = 0
#        m = len(args)
#        for i in range(m):
#            ret += args[i] * pow(x, (m-i-1))
#        return ret
#    length = len(args)
#    for i in range(length):
#        if 0 == length - i - 1:
#            ret_str += "_"
#        else:
#            ret_str += "_*x^" + str(length-i-1) + "+"
#    return poly, ret_str


def polynomial(model, length, new_length):
    args = construct_params(length+new_length)
    if model == None:
        def poly(x, *args):
            ret = 0
            m = len(args)
            for i in range(m):
                ret += args[i] * x.pow(m-i-1)
            return ret
    else:
        def poly(x, *args):
            model_args = args[:length]
            cur_args = args[-new_length:]
            ret = 0
            m = len(cur_args)
            new_input = model(x, *model_args)
            for i in range(m):
                ret += cur_args[i] * new_input.pow(m-i-1)
            return ret

    return poly



def exponential(model, length, new_length):
    args = construct_params(length+new_length)
    if model == None:
        def exp(x, *args):
            return args[0]*torch.exp(args[1]*x+args[2])
    else:
        def exp(x, *args):
            model_args = args[:length]
            cur_args = args[-new_length:]
            new_input = model(x, *model_args)
            return cur_args[0]*torch.exp(cur_args[1]*new_input+cur_args[2])

    return exp


def logarithm(model, length, new_length):
    args = construct_params(length+new_length)
    if model == None:
        def log(x, *args):
            return args[0] * torch.log(args[1] * x + args[2])
    else:
        def log(x, *args):
            model_args = args[:length]
            cur_args = args[-new_length:]
            new_input = model(x, *model_args)
            return cur_args[0] * torch.log(cur_args[1] * new_input + cur_args[2])

    return log


def add_exponential(model, length, new_length):
    args = construct_params(length+new_length)
    def func(x, *args):
        model_args = args[:length]
        cur_args = args[-new_length:]
        new_input = model(x, *model_args)
        return new_input + cur_args[0] * torch.exp(cur_args[1] * x + cur_args[2])
    return func


def add_logarithm(model, length, new_length):
    args = construct_params(length+new_length)
    def func(x, *args):
        model_args = args[:length]
        cur_args = args[-new_length:]
        new_input = model(x, *model_args)
        return new_input + cur_args[0] * torch.log(cur_args[1] * x + cur_args[2])

    return func

def add_polynomial(model, length, new_length):
    args = construct_params(length+new_length)
    def func(x, *args):
        model_args = args[:length]
        cur_args = args[-new_length:]
        ret = 0
        m = len(cur_args)
        new_input = model(x, *model_args)
        for i in range(m):
            ret += cur_args[i] * x.pow(m-i-1)
        return new_input + ret

    return func

def inverse(model, length, new_length):
    args = construct_params(length+new_length)
    def func(x, *args):
        model_args = args[:length]
        cur_args = args[-new_length:]
        new_input = model(x, *model_args)
        return cur_args[0] / new_input

    return func


def construct_params(length):
    ret = []
    for i in range(length):
        ret.append("params_"+str(i))
    return ret



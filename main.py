import os
import numpy as np
import random as rand
import functions
import test_func
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy


MAX_LENGTH = 5
MAX_POLY_LENGTH = 5
func_zoo = ["polynomial", "exponential", "logarithm", "add_exponential", "add_logarithm","add_polynomial", "inverse"]
MAX_COUNT = 500
class MathModel:

    def __init__(self, model_length, init_poly=False, poly_num = None, init_log = False, init_exp = False, func_list = None):

        assert model_length >= 1
        self.model_length = model_length

        if func_list == None:
            if init_poly:
                assert poly_num <= MAX_POLY_LENGTH
                self.func_list = self.init_function(poly = True,polynum = poly_num)
            elif init_log:
                self.func_list = self.init_function(log = True)
            elif init_exp:
                self.func_list = self.init_function(exp = True)
            else:
                self.func_list = self.construct_list()
        else:
            assert model_length == len(func_list)
            self.func_list = func_list

        self.generate_model()

    def __eq__(self, other):
        if self.model_length != other.model_length:
            return False
        #print(self.model_length)
        #print(other.model_length)
        #print(self.func_list)
        #print(other.func_list)
        for i in range(self.model_length):
            if self.func_list[i] != other.func_list[i]:
                return False

        return True

    def __lt__(self, other):
        return self.error < other.error


    def construct_list(self):
        ret = self.init_function()
        for i in range(self.model_length-1):
            func = func_zoo[rand.randint(0,6)]
            if func == "polynomial" or func == "add_polynomial":
                poly_length = rand.randint(1, MAX_POLY_LENGTH)
                ret.append(func+str(poly_length))
            else:
                ret.append(func)

        return ret


    def init_function(self, poly = False, polynum = None, log = False, exp = False):
        if poly:
            init_func = "polynomial"+str(polynum)
        elif log:
            init_func = "logarithm"
        elif exp:
            init_func = "exponential"
        else:
            init_func = func_zoo[rand.randint(0,2)]
            if init_func == "polynomial":
                poly_length = rand.randint(1, MAX_POLY_LENGTH)
                return [init_func+str(poly_length)]
        return [init_func]


    def generate_model(self):
        model = None
        paramcount = 0
        for func in self.func_list:
            if func.startswith("polynomial") or func.startswith("add_polynomial"):
                length = int(func[-1])
                #print(func)
                func = func[:-1]
                #print(func)
                mathmodel = getattr(functions,func)
                model = mathmodel(model, paramcount, length)
                paramcount += length
            elif func == "inverse":
                mathmodel = getattr(functions,func)
                model = mathmodel(model, paramcount, 1)
                paramcount += 1
            else:
                mathmodel = getattr(functions,func)
                model = mathmodel(model, paramcount, 3)
                paramcount += 3

        self.model = model
        self.paramcount = paramcount


    def print_function(self):
        cur_str = ""
        for i, func in enumerate(self.func_list):
            if func.startswith("polynomial"):
                length = int(func[-1])
                cur_str += "y = "
                if i == 0:
                    cur_str += poly_str(length, "x")
                else:
                    cur_str += poly_str(length, "y")
                cur_str += "\n"
            elif func.startswith("exp"):
                cur_str += "y = "
                if i == 0:
                    cur_str += "_*exp(_ * x + _)"
                else:
                    cur_str += "_*exp(_ * y + _)"
                cur_str += "\n"
            elif func.startswith("log"):
                cur_str += "y = "
                if i == 0:
                    cur_str += "_*log(_ * x + _)"
                else:
                    cur_str += "_*log(_ * y + _)"
                cur_str += "\n"
            elif func.startswith("add_polynomial"):
                length = int(func[-1])
                cur_str += "y = y + "
                cur_str += poly_str(length, "x")
                cur_str += "\n"
            elif func.startswith("add_log"):
                cur_str += "y = y + "
                cur_str += "_*log(_ * x + _)"
                cur_str += "\n"
            elif func.startswith("add_exp"):
                cur_str += "y = y + "
                cur_str += "_*exp(_ * x + _)"
                cur_str += "\n"
            elif func.startswith("inverse"):
                cur_str += "y = _ / y"
                cur_str += "\n"
        self.model_strings = cur_str
        self.replace_parameter()
        print(self.model_strings)
        print("MSE: ", self.error)
        return self.model_strings, self.error


    def replace_parameter(self):
        for i in range(len(self.params)):
            self.model_strings = self.model_strings.replace("_", "("+str(self.params[i])+")", 1)



    def fit(self, x, y):
        try:
            for i in range(3):
                fittedParams, pcov = curve_fit(self.model, x, y, np.ones(self.paramcount),maxfev = 10000)
                modelPrediction = self.model(x, *fittedParams)
                absError = np.abs(modelPrediction - y)
                mse = np.mean(absError)
                if i == 0:
                    error = mse
                    bestparam = fittedParams
                if(mse < error):
                    error = mse
                    bestparam = fittedParams

            self.params = bestparam

            self.error = error

            return error
        except RuntimeError:
            self.error = np.inf
            self.params = np.ones(self.paramcount)
            return self.error

def poly_str(length, token):
    ret_str = ""
    for i in range(length):
        if 0 == length - i - 1:
            ret_str += "_"
        else:
            ret_str += "_*" + token+ "^" + str(length-i-1) + "+"
    return ret_str



def findMaxPower(x, y):
    xmin = np.abs(x[np.argmin(np.abs(y))])
    xmax = np.abs(x[np.argmax(np.abs(y))])
    ymin = np.min(np.abs(y))
    ymax = np.max(np.abs(y))

    xdiff = xmax-xmin
    ydiff = ymax-ymin

    for i in range(100):
        if xdiff < 1:
            if ydiff < 1:
                if ydiff > xdiff**i:
                    if i > MAX_POLY_LENGTH:
                        return MAX_POLY_LENGTH
                    return i
            else:
                return MAX_POLY_LENGTH
        #add small momentum
        if xdiff == 1:
            xdiff += 0.1
        if ydiff < xdiff**i:
            if i > MAX_POLY_LENGTH:
                return MAX_POLY_LENGTH
            return i
    return MAX_POLY_LENGTH


def construct_params(length):
    ret = []
    for i in range(length):
        ret.append("params_"+str(i))
    return ret


class synthesizer:
    def __init__(self,training_input, training_output, bound):
        self.input=training_input
        self.output=training_output
        self.model_population = []
        self.errorbound = bound

    def search(self):
        if self.search_linear():
            return
        if self.search_log():
            return
        if self.search_exp():
            return

        for i in range (10):
            length = rand.randint(2, MAX_LENGTH)
            new_func = MathModel(length)
            if new_func not in self.model_population:
                new_func.fit(self.input, self.output)
                self.model_population.append(new_func)


        self.model_population.sort()

        search_count = 0
        while self.model_population[0].error > self.errorbound:
            self.genetic()
            self.model_population.sort()
            search_count += 1
            print(search_count)
            if search_count >= MAX_COUNT:
                break


        self.model = self.model_population[0]
        return


    def genetic(self):
        length = len(self.model_population)
        if length > 3:
            for i in range(length - 3):
                self.model_population.pop()

        new_length = len(self.model_population)
        for i in range(new_length):
            for j in np.arange(i+1, new_length):
                if self.model_population[i].model_length < 2 or self.model_population[j].model_length < 2:
                    continue
                new_model1, new_model2 = self.cross(i, j)

                if new_model1 not in self.model_population:
                    new_model1.fit(self.input, self.output)
                    self.model_population.append(new_model1)

                if new_model2 not in self.model_population:
                    new_model2.fit(self.input, self.output)
                    self.model_population.append(new_model2)

        for i in range(new_length):
            new_model = self.mutate(i)
            if new_model not in self.model_population:
                new_model.fit(self.input, self.output)
                self.model_population.append(new_model)


    def mutate(self, i):
        old_model = copy.deepcopy(self.model_population[i])
        model_length = old_model.model_length
        old_funcs = old_model.func_list
        if model_length <=2:
            mutate_type = ["add", "change"]
        elif model_length < MAX_LENGTH:
            mutate_type = ["add", "minus", "change"]
        else:
            mutate_type = ["minus", "change"]

        cur_type = mutate_type[rand.randint(0, len(mutate_type)-1)]
        if cur_type == "add":
            ins_pos = rand.randint(1, model_length)
            ins_function = func_zoo[rand.randint(0,6)]
            if ins_function == "polynomial" or ins_function == "add_polynomial":
                poly_length = rand.randint(1, MAX_POLY_LENGTH)
                ins_function += str(poly_length)
            old_funcs.insert(ins_pos, ins_function)
        elif cur_type == "minus":
            rem_pos = rand.randint(1, model_length-1)
            old_funcs.pop(rem_pos)
        else:
            ch_pos = rand.randint(0, model_length-1)
            if ch_pos == 0:
                ch_function = func_zoo[rand.randint(0,2)]
            else:
                ch_function = func_zoo[rand.randint(0,6)]
            if ch_function == "polynomial" or ch_function == "add_polynomial":
                poly_length = rand.randint(1, MAX_POLY_LENGTH)
                ch_function += str(poly_length)
            old_funcs[ch_pos] = ch_function

        new_model = MathModel(model_length = len(old_funcs), func_list = old_funcs)
        return new_model

    def cross(self, i, j):
        first_model = copy.deepcopy(self.model_population[i])
        second_model = copy.deepcopy(self.model_population[j])
        first_funcs = first_model.func_list
        first_length = len(first_funcs)
        second_funcs = second_model.func_list
        second_length = len(second_funcs)


        new_funcs1 = first_funcs[:first_length//2]+second_funcs[second_length//2:]
        new_funcs2 = second_funcs[:second_length//2]+first_funcs[first_length//2:]


        new_model1 = MathModel(model_length = len(new_funcs1), func_list = new_funcs1)
        new_model2 = MathModel(model_length = len(new_funcs2), func_list = new_funcs2)

        return new_model1, new_model2


    def search_log(self):
        model = MathModel(1, init_log=True)
        error = model.fit(self.input, self.output)
        if (error <= self.errorbound):
            self.error = error
            self.model = model
            return True
        return False


    def search_exp(self):
        model = MathModel(1, init_exp=True)
        error = model.fit(self.input, self.output)
        if (error <= self.errorbound):
            self.error = error
            self.model = model
            return True
        return False


    def search_linear(self):
        power=findMaxPower(self.input,self.output)
        for i in np.arange(power,0, -1):
            model = MathModel(1, init_poly=True, poly_num=power)
            error = model.fit(self.input, self.output)
            if (error <= self.errorbound):
                self.error = error
                self.model = model
                return True
        return False


    def clean_params(self):
        for i in range(self.paramscount):
            if np.abs(self.paramsvals[i]) < 10**(-4):
                self.paramsvals[i]=0
            if np.abs(round(self.paramsvals[i]) - self.paramsvals[i]) < 10**(-5):
                self.paramsvals[i] = round(self.paramsvals[i])




def generate_data(n, test_func):
    x = []
    y = []
    for i in range(n):
        newx = rand.random()*rand.randint(0,50)
        newy = test_func(newx)+ np.random.normal(0,0.2,1)[0]
        if (np.isinf(newy)):
            continue
        x.append(newx)
        y.append(newy)

    return np.array(x), np.array(y)

def plot(test_func, index):
    test_string = "test_func" + str(index)
    rand.seed(1825)
    x, y=generate_data(200, test_func)
    with open(test_string + "_x", "w") as f:
        for item in x:
            f.write("%s\n" % item)
    with open(test_string + "_y", "w") as f:
        for item in y:
            f.write("%s\n" % item)
    x = np.repeat(x,200)
    y = np.repeat(y, 200)
    #model = MathModel(1)
    #print(model.paramcount)
    #print(model.func_list)
    #mse = model.fit(x,y)
    #print(mse)
    #print(model.params)
    syns = synthesizer(x, y, 0.5)
    syns.search()
    model_string, error = syns.model.print_function()
    with open("output.txt", "a+") as f:
        f.write(model_string)
        f.write("\n")
        f.write("error: " + str(error) + "\n")
    f = plt.figure()
    axes = f.add_subplot(111)

    axes.plot(x, y, 'D', color = "green", label = "Data")

    xModel = np.linspace(min(x), max(x))
    yModel = syns.model.model(xModel, *syns.model.params)
    realY = test_func(xModel)

    axes.plot(xModel, realY, color = "blue", label = "FittedModel" )

    axes.plot(xModel, yModel, color = "red", label = "RealModel")

    axes.legend(loc = "upper left")

    axes.set_xlabel('X Data')
    axes.set_ylabel('Y Data')

    plt.savefig(test_string+".png")
    plt.close('all')

def main():
    for i in range(8):
        tests = getattr(test_func, "test_func"+str(i+1))
        plot(tests, i+1)


if __name__ == "__main__":
    main()


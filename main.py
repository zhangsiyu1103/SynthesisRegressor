import os
import numpy as np
import random as rand
import functions
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



MAX_LENGTH = 5
MAX_POLY_LENGTH = 5
func_zoo = ["polynomial", "exponential", "logarithm", "add_exponential", "add_logarithm","add_polynomial", "inverse"]

class MathModel:

    def __init__(self, model_length, init_poly=False, poly_num = None, init_log = False, init_exp = False, func_list = None):

        assert model_length >= 1
        self.model_length = model_length

        if func_list == None:
            if init_poly:
                self.func_list = self.init_function(poly = True,polynum = poly_num)
            elif init_log:
                self.func_list = self.init_function(log = True)
            elif init_exp:
                self.func_list = self.init_function(exp = True)
            else:
                self.func_list = self.construct_list()
        else:
            self.func_list = func_list

        self.generate_model()

    def __eq__(self, other):
        if self.model_length != other.model_length:
            return False

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
                func = func[:-1]
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


    def replace_parameter(self):
        for i in range(len(self.params)):
            self.model_strings = self.model_strings.replace("_", "("+str(self.params[i])+")", 1)



    def fit(self, x, y):
        try:
            for i in range(5):
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
                    return i
            else:
                return 5
        #add small momentum
        if xdiff == 1:
            xdiff += 0.1
        if ydiff < xdiff**i:
            return i
    return 100


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
        while self.model_population[0].error > self.errorbound:
            self.genetic()
            self.model_population.sort()


        self.model = self.model_population[0]
        return


    def genetic(self):
        length = len(self.model_population)
        for i in range(length//2):
            self.model_population.pop()

        new_length = len(self.model_population)
        for i in range(new_length):
            for j in np.arange(i+1, new_length):
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
        old_model = self.model_population[i]
        old_funcs = old_model.func_list
        if old_model.model_length < MAX_LENGTH:
            mutate_type = ["add", "minus", "change"]
        elif old_model.model_lenth <=2:
            mutate_type = ["add", "change"]
        else:
            mutate_type = ["minus", "change"]

        cur_type = mutate_type[rand.randint(0, len(mutate_type)-1)]
        if cur_type == "add":
            ins_pos = rand.randint(1,self.model_length)
            ins_function = func_zoo[rand.randint(0,6)]
            old_funcs.insert(ins_pos, ins_function)
        elif cur_type == "minus":
            rem_pos = rand.randint(1,self.model_length-1)
            old_funcs.pop(ins_pos, ins_function)
        else:
            ch_pos = rand.randint(0,self.model_length-1)
            if ch_pos == 0:
                ch_function = func_zoo[rand.randint(0,2)]
            else:
                ch_function = func_zoo[rand.randint(0,6)]
            old_funcs[ch_pos] = ch_function

        new_model = MathModel(model_length = len(old_funcs), func_list = old_funcs)
        return new_model

    def cross(self, i, j):
        first_model = self.model_population[i]
        second_model = self.model_population[j]
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
            model = MathModel(1, init_poly=True, poly_num=power+1)
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



def test_func(x):
    return np.log(5*x+3) + 2*x +4

def generate_data(n):
    x = []
    y = []
    for i in range(n):
        newx = rand.random()*rand.randint(0,50)
        x.append(newx)
        y.append(test_func(newx))
                #+ np.random.normal(0,0.5,1)[0])

    return np.array(x), np.array(y)

def main():
    rand.seed(1825)
    x, y=generate_data(200)
    x = np.repeat(x,200)
    y = np.repeat(y, 200)
    #model = MathModel(1)
    #print(model.paramcount)
    #print(model.func_list)
    #mse = model.fit(x,y)
    #print(mse)
    #print(model.params)
    syns = synthesizer(x, y, 0.1)
    syns.search()
    syns.model.print_function()
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

    plt.savefig("demo.png")
    plt.close('all')


if __name__ == "__main__":
    main()


from numpy import random
import numpy
import math


def id_gen():
    names = ["P", "Q", "R"]
    for n in names:
        yield n


ids = id_gen()


class Model:

    def __init__(self, input_count, output_count, learn_rate, debug=False):
        self.outputs = []
        self.debug = debug
        self.learn_rate = learn_rate
        for i in range(output_count):
            self.outputs.append(Vector(input_count, next(ids)))

    def train(self, data_set):
        done = False
        first_loop = True
        old = {}
        c = 0
        while not done:
            c += 1
            groups = {}
            '''This section actually trains the data'''
            for name in data_set:
                inputs = data_set[name]
                output = self.cycle(inputs)
                groups[name] = output

            '''Run at least two loops to see if anything moves'''
            if first_loop:
                for n in groups:
                    old[n] = groups[n]
                first_loop = False
                continue

            '''And now we check to see if everything is still being assigned to the same vector.
            If everything is, running further training is just a waste of CPU for this training data,
            if it's not, we need to run it again.'''
            done = True
            for n in groups:
                if old[n] != groups[n]:
                    done = False
            '''We also need to update the old list'''
            for n in groups:
                old[n] = groups[n]

        if self.debug:
            for o in self.outputs:
                print(o.id, o.print())

    def cycle(self, inputs):
        v = max(self.outputs, key=lambda vector: vector.output(inputs))
        v.learn(inputs, self.learn_rate)
        return v.id

    def run(self, inputs, debug=False):
        for i in inputs.keys():
            in_set = inputs[i]
            v = max(self.outputs, key=lambda vector: vector.output(in_set))
            if debug:
                totals = [(x.id, round(x.output(in_set), 4)) for x in self.outputs]
                print(f"{i} :\t{v.id}:\t{totals}")
            else:
                print(f"{i} :\t{v.id}")


class Vector:
    def __init__(self, input_count, id):
        self.id = id
        self.values = []
        for i in range(input_count):
            r = random.rand()
            self.values.append(r)
            self.normalize()

    def normalize(self):
        length = math.sqrt(sum(v ** 2 for v in self.values))
        self.values = list(map(lambda x: x / length, self.values))

    def output(self, input_set):
        return sum(numpy.multiply(self.values, input_set))

    def learn(self, input_set, learn_rate):
        adjustments = numpy.array(input_set) * learn_rate
        self.values = numpy.add(self.values, adjustments)
        self.normalize()

    def print(self):
        return str([round(x, 4) for x in self.values])


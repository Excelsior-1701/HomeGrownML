import numpy as np
from decimal import *  # Floating point errors in ML are bad, mkay?


class PerceptionModel:

    def __init__(self, weights, learning_coefficient, threshold, func, debug=False):
        self.target_func = func
        self.learning_coefficient = Decimal(str(learning_coefficient))

        self.weights = []
        for w in weights:
            self.weights.append(Decimal(str(w)))

        self.threshold = Decimal(str(threshold))
        self.trained = False
        self.debug = debug

        # This is honestly easier to just precalculate and save right now, though this obviously won't scale super well
        self.input_sets = []
        inputs = 2 ** len(weights)
        for i in range(inputs):
            variable_count = len(weights)
            in_set = list(bin(i)[2:].zfill(variable_count))
            in_set = np.array(in_set)
            s = []
            for j in in_set:
                s.append(Decimal(j))
            self.input_sets.append(s)

    def get_weights(self):
        s = []
        for w in self.weights:
            s.append(str(w))
        return str(s)

    def train(self, max_iterations=0, detect_loops=False):
        if self.debug:
            print(f"{'epoch':<7} {'Inputs':<20} {'Result':<10} {'Target':<10} {'Old Weights':<40} {'New Weights':<40}")
        done = False
        iteration = 0
        if detect_loops:
            history = []
        while not done:
            iteration += 1
            done = self.epoch(iteration)
            if done:
                break
            if max_iterations != 0 and iteration >= max_iterations:
                break
            if detect_loops:
                if str(self.get_weights()) in history:
                    print("Infinite loop detected")
                    break
                else:
                    history.append(str(self.get_weights()))

        if done:
            self.trained = True

        if self.debug:
            print("Final Weights: " + str(self.get_weights()))

    def epoch(self, iteration):
        done = True
        for attributes in self.input_sets:
            if self.debug:
                old_weights = self.get_weights()

            result = self.check(attributes)
            target = self.target_func(attributes)

            if target != result:
                done = False

            a = []
            for i in range(len(attributes)):
                adj_weight = self.weights[i] + self.learning_coefficient * Decimal(str(target - result)) * attributes[i]
                self.weights[i] = adj_weight
                a.append(str(attributes[i]))

            if self.debug:
                new_weights = self.get_weights()
            if self.debug:
                print(f"{iteration:<7} {str(a):<20} {result:<10} {target:<10} {old_weights:<40} {new_weights:<50}")

        return done

    '''Internal function to handle checking a single input'''
    def check(self, inputs):
        products = np.multiply(self.weights, inputs)
        total = np.sum(products)
        result = (Decimal(str(total)) >= self.threshold)
        if result:
            return 1
        return 0

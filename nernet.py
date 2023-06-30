from numpy import random
import inputsets
import math


def id_gen():
    n = 0
    while True:
        yield n
        n += 1


ids = id_gen()


class Node:
    def __init__(self, output_func):
        self.output_func = output_func
        self.value = 0
        self.links_in = []
        self.links_out = []
        self.id = next(ids)

    def get_output(self):
        return self.output_func(self.value)

    def add_link_in(self, link):
        self.links_in.append(link)

    def add_link_out(self, link):
        self.links_out.append(link)


class InputNode(Node):
    def __init__(self):
        super().__init__(lambda x: float(x))
        self.links_in = []
        self.links_out = []

    def set_value(self, v):
        self.value = v

    def run(self):
        pass

    def set_delta(self):
        pass


class OutputNode(Node):
    def __init__(self, output_func):
        super().__init__(output_func)
        self.delta = 0

    def run(self):
        ins = map(lambda l: l.get_value(), self.links_in)
        total = sum(ins)
        self.value = total

    def set_delta(self, target_output):
        actual_output = self.get_output()
        delta = (target_output - actual_output) * actual_output * (1 - actual_output)
        self.delta = delta


class HiddenNode(Node):
    def __init__(self, output_func):
        super().__init__(output_func)
        self.delta = 0

    def run(self):
        ins = list(map(lambda l: l.get_value(), self.links_in))
        total = sum(ins)
        self.value = total

    def set_delta(self):
        actual_output = self.get_output()
        total = sum(map(lambda link: link.weight * link.node_out.delta, self.links_out))
        delta = total * actual_output * (1 - actual_output)
        self.delta = delta


class Link:
    def __init__(self, node_in, node_out, learn_rate, weight):
        self.node_in = node_in
        self.node_out = node_out
        self.weight = weight
        self.learn_rate = learn_rate
        self.id = next(ids)

    def update_weight(self):
        out_delta = self.node_out.delta
        prev_out = self.node_in.get_output()
        new_weight = self.weight + self.learn_rate * out_delta * prev_out
        self.weight = new_weight

    def get_value(self):
        return self.node_in.get_output() * self.weight


class Network:
    def __init__(self, layers, learn_rate, variance, output_func=None):
        if output_func is None:
            def sigmoid(x):
                return 1 / (1 + math.exp(-x))
            output_func = sigmoid
        self.variance = variance
        self.layers = []
        self.links = []
        self.cycles = 0
        """First, we create the actual nodes"""
        in_layer = []
        for i in range(layers[0]):
            n = InputNode()
            in_layer.append(n)
        self.layers.append(in_layer)

        for i in layers[1:-1]:
            hidden_layer = []
            for j in range(i):
                n = HiddenNode(output_func)
                hidden_layer.append(n)
            self.layers.append(hidden_layer)

        out_layer = []
        for i in range(layers[-1]):
            n = OutputNode(output_func)
            out_layer.append(n)
        self.layers.append(out_layer)
        """Now create the links between them"""
        for in_layer_number in range(len(self.layers) - 1):
            in_nodes = self.layers[in_layer_number]
            out_nodes = self.layers[in_layer_number + 1]
            for p in in_nodes:
                for q in out_nodes:
                    x = Link(p, q, learn_rate, (random.rand() - .5) * 2)
                    p.add_link_out(x)
                    q.add_link_in(x)
                    self.links.append(x)

    def forward_pass(self, inputs):
        """Feed inputs in"""
        for i in range(len(inputs)):
            self.layers[0][i].set_value(inputs[i])

        """Propagate them forward"""
        for layer in self.layers[1:]:
            for node in layer:
                node.run()

        """Store outputs"""
        outs = []
        for output_node in self.layers[-1]:
            outs.append(output_node.get_output())
        return outs

    def backward_pass(self, targets):
        for i in range(len(self.layers[-1])):
            node = self.layers[-1][i]
            output = targets[i]
            node.set_delta(output)

        for layer in reversed(self.layers[1:-1]):
            for node in layer:
                node.set_delta()

        for link in self.links:
            link.update_weight()

    def cycle(self, target_output_func):
        done = True
        input_set = inputsets.get_input_set(len(self.layers[0]))
        err = 0
        for i in input_set:
            t_outs = target_output_func(i)
            results = self.forward_pass(i)
            self.backward_pass(t_outs)
            for j in range(len(t_outs)):
                if abs(t_outs[j] - results[j]) > self.variance:
                    done = False
            err += sum(map(lambda x, y: .5 * ((x - y) ** 2), t_outs, results))
            new_results = self.forward_pass(i)
            print("Inputs:\t" + str(i) + "\tResults:\t" + str(results) + "\tTargets:\t" + str(t_outs) + "\tChanged To:\t" + str(new_results))
        print("Error:\t" + str(err))
        return done

    def train(self, target_output_func):
        while not self.cycle(target_output_func):
            pass

# THIS FILE BUILDS NEURAL NETWORKS FROM THE SCALAR ENGINE.
# engine.py supplies numbers that remember operations and derivatives.
# Here we arrange those numbers into neurons, layers, and a network.
# These are class definitions: no dataset or training loop runs on its own.
# Import Python's standard random module so we can initialize weights.
import random

# Load the Value class from engine.py, making the name Value available
# here. When nn.py is run directly, its own folder is on Python's import
# search path. This import expects engine to be available on that path.
from engine import Value

# A SHARED BASE CLASS: neurons, layers, and networks all have parameters
# and need to clear gradients. Put shared behavior here instead of repeating
# it. A class defines behavior; creating an object makes a particular instance.
class Module:

    # Call model.zero_grad() before a new backward pass. Python passes model
    # as self automatically, so you supply no argument. A method is a function
    # attached to a class. Here self can be a neuron, layer, or entire MLP.
    def zero_grad(self):

        # Ask THIS object's parameters() method for its trainable Values.
        # Python uses the method belonging to the actual object's class. Thus an
        # MLP gets all its layers' parameters even though zero_grad is defined here.
        # The for loop takes one parameter at a time and temporarily calls it p.
        for p in self.parameters():

            # Reset the gradient stored on that actual Value object. We do not reset
            # p.data, so learned weights stay intact. Backpropagation uses +=, making
            # this reset necessary to avoid carrying previous steps' parameter gradients.
            p.grad = 0

    # Return a list of the trainable Value objects owned by this component.
    # The objects themselves are returned, not copies of their data: callers
    # can reset their gradients or update their numbers through this list.
    def parameters(self):

        # A plain Module owns no parameters, so return an empty list.
        # Subclasses below replace this method with their own implementations.
        return []

# ONE NEURON: compute bias + sum(weight*input), optionally followed by
# ReLU. Neuron(Module) means inheritance: a Neuron gets Module's methods,
# such as zero_grad, unless it defines its own version.
class Neuron(Module):

    # INITIALIZE A NEURON: Neuron(3) creates an object, passes it as self,
    # passes 3 as nin, and uses the default nonlin=True. You never manually
    # pass self. Neuron(3, nonlin=False) instead creates a linear neuron.
    # nin is the number of input features; nonlin controls the activation.
    def __init__(self, nin, nonlin=True):

        # Create a list of nin separate trainable weights. This compact syntax
        # is a list comprehension: repeat Value(random.uniform(-1,1)) nin times.
        # range(nin) gives 0 through nin-1; _ holds each index, but we do not use it.
        # uniform draws a random number between -1 and 1. Wrapping it in Value
        # lets later operations compute gradients with respect to that weight.
        # self.w stores the resulting list on this particular neuron.
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]

        # Create one trainable bias with initial numerical value zero.
        # Each neuron gets its own separate Value object for this bias.
        self.b = Value(0)

        # Remember the activation choice on this neuron so its forward method
        # can use the same choice whenever the neuron is called.
        self.nonlin = nonlin

    # FORWARD CALCULATION: defining __call__ makes an object callable.
    # For neuron([2,3]), Python calls neuron.__call__([2,3]): self is the
    # neuron object, and x is the input list. Creating a neuron initializes
    # its parameters; calling that existing neuron computes an output.
    def __call__(self, x):

        # Compute the weighted sum. zip(self.w, x) pairs weight 0 with input 0,
        # weight 1 with input 1, and so on. wi,xi unpacks each pair into two names.
        # The generator expression wi*xi for ... produces one product at a time.
        # sum(products, self.b) starts with the bias, then adds those products.
        # Because weights and bias are Values, these operations build a graph.
        # zip stops at the shorter sequence; this code does not check input length.
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)

        # "A if condition else B" selects a result. If nonlin is True, return
        # ReLU of the weighted sum; otherwise return the weighted sum unchanged.
        # Both choices return a Value connected to the calculation's inputs.
        return act.relu() if self.nonlin else act

    # Return a list of the trainable Value objects owned by this component.
    # The objects themselves are returned, not copies of their data: callers
    # can reset their gradients or update their numbers through this list.
    def parameters(self):

        # Here + joins lists, rather than doing numerical addition. [self.b]
        # is a one-element list, so the result contains every weight and the bias.
        # The returned list is new, but it refers to the same parameter objects.
        return self.w + [self.b]

    # Return a readable description of this object when Python displays it.
    # This method constructs text only; it does not print anything or train.
    def __repr__(self):

        # An f-string evaluates the expressions inside braces. The conditional
        # chooses the activation name, and len(self.w) counts the input weights.
        # For example, the resulting text might be ReLUNeuron(3).
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

# ONE LAYER: several neurons independently read the same input list.
# Layer inherits zero_grad from Module and supplies its own parameters
# method so clearing gradients reaches every neuron in the layer.
class Layer(Module):

    # INITIALIZE A LAYER: Layer(3,4,nonlin=False) means three inputs and
    # four output neurons. self is supplied automatically, nin=3, nout=4.
    # **kwargs collects extra named arguments into a dictionary, for example
    # {'nonlin': False}. Here ** means keyword collection, not exponentiation.
    def __init__(self, nin, nout, **kwargs):

        # Create nout different neurons, each with nin inputs. Each iteration
        # creates fresh weights and a fresh bias. In this call, **kwargs UNPACKS
        # the dictionary into named arguments, forwarding nonlin=False if provided.
        # Store these neuron objects in a list on this layer.
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    # LAYER FORWARD PASS: layer(x) invokes this method automatically.
    # self is this Layer object; x is the input list shared by its neurons.
    def __call__(self, x):

        # For every neuron n in the layer, call n(x) using the same input x.
        # Collect its returned Value in a list. A layer with four neurons produces
        # four outputs; each one still remembers its own computation graph.
        out = [n(x) for n in self.neurons]

        # Python indexing starts at zero, so out[0] is the first output.
        # len(out) counts outputs. If there is only one, return that Value directly;
        # otherwise return the whole list. Consequently, single-output layers
        # return a different type from multiple-output layers.
        return out[0] if len(out) == 1 else out

    # Return a list of the trainable Value objects owned by this component.
    # The objects themselves are returned, not copies of their data: callers
    # can reset their gradients or update their numbers through this list.
    def parameters(self):

        # Flatten all neurons' parameter lists into one list. Read this as:
        # for each neuron n, get n.parameters(); then include each parameter p.
        # This is a nested list comprehension, equivalent to two nested loops.
        return [p for n in self.neurons for p in n.parameters()]

    # Return a readable description of this object when Python displays it.
    # This method constructs text only; it does not print anything or train.
    def __repr__(self):

        # str(n) asks for each neuron's text description. ', '.join(...) joins
        # those strings with commas between them. The surrounding f-string adds
        # the layer label and brackets; this only returns a description.
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

# A WHOLE MULTILAYER PERCEPTRON: store layers in order and pass an input
# through them. MLP inherits zero_grad from Module, just like a neuron.
# This class computes predictions but does not define a loss or optimizer.
class MLP(Module):

    # INITIALIZE THE NETWORK: MLP(3, [4,4,1]) supplies nin=3 and
    # nouts=[4,4,1]; Python supplies self automatically. nouts is a list of
    # output widths, one per layer, including the final output layer.
    def __init__(self, nin, nouts):

        # Join [nin] and nouts to make a list of all widths. In this example,
        # sz=[3,4,4,1]. Adjacent entries describe input/output widths for a layer.
        # sz is a local variable used while building this network.
        sz = [nin] + nouts

        # Create one Layer per entry in nouts. range(len(nouts)) gives each
        # layer index i, starting at zero. sz[i] is its input width and sz[i+1]
        # is its output width. != means "not equal". len(nouts)-1 is the last
        # index, so nonlin is True for hidden layers and False for the final one.
        # For MLP(3,[4,4,1]), build 3->4 ReLU, 4->4 ReLU, and 4->1 linear.
        # The code assumes layer widths can be connected under Layer's return
        # convention; it does not validate architectures or input shapes.
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    # NETWORK FORWARD PASS: model(x) invokes this method automatically.
    # self is this MLP object; x initially contains the input features.
    # Every call computes fresh intermediate Values and a fresh graph.
    def __call__(self, x):

        # Visit layers in their stored order, starting with the first.
        # The same layer objects and their weights are reused for each prediction.
        for layer in self.layers:

            # Call this layer with the current input and assign its output to the
            # local name x. On the next loop iteration, x is the next layer's input.
            # Reassigning this name does not erase earlier Values: the computation
            # graph still holds references to the inputs needed for backpropagation.
            x = layer(x)

        # After the final layer, return its output: a single Value for a
        # one-neuron output layer, or a list of Values for a wider output layer.
        return x

    # Return a list of the trainable Value objects owned by this component.
    # The objects themselves are returned, not copies of their data: callers
    # can reset their gradients or update their numbers through this list.
    def parameters(self):

        # Collect every layer's trainable Values into a single flat list.
        # This lets model.zero_grad() reach all parameters. A training loop can
        # also iterate over this list and update p.data using p.grad.
        return [p for layer in self.layers for p in layer.parameters()]

    # Return a readable description of this object when Python displays it.
    # This method constructs text only; it does not print anything or train.
    def __repr__(self):

        # Ask each layer for its text description and join those descriptions
        # with commas. Return the full network description; no logging happens.
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

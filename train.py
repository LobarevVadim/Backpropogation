# HOW THIS FILE WORKS
# This script teaches a small neural network the XOR pattern: two matching
# inputs should give -1, and two different inputs should give +1.
# It creates the network once, then repeats: predict, calculate error,
# clear old gradients, backpropagate, and update weights and biases.
# engine.py handles arithmetic and derivatives; nn.py builds the network.
# This file supplies the examples and the training loop. No extra packages
# are required. Training starts when this file is executed (or imported).
# 
# Import Python's random-number module. nn.py uses the same module to
# initialize weights. A trailing semicolon is optional in Python.
import random; 

# Import the MLP class from nn.py. A class is a blueprint for objects;
# importing it does not yet create a network. Running this script puts its
# folder on the import search path, allowing Python to find nn.py.
from nn import MLP; 

# Make random initialization repeatable. 42 is an arbitrary seed, not a
# learning rate or network size. With the same code and environment, each
# run begins with the same random weights, making results easier to compare.
random.seed(42)

# Create the actual network and store it in model. MLP(2, [4,1]) means
# 2 input features, a hidden layer of 4 ReLU neurons, and 1 linear output.
# Python automatically passes the new object as self to MLP.__init__;
# you supply only the input width and list of layer widths.
# Construction creates layers, neurons, and their Value weights and biases.
# There are 17 trainable parameters: 4*(2 weights + 1 bias) + 4 weights + 1 bias.
model = MLP(2, [4, 1]); 

# Store the four training examples in a list (the outer square brackets).
# Each tuple, written (...), pairs an input list with its expected output:
#   ([-1,-1],-1) means inputs -1 and -1 should produce -1.
#   ([-1, 1], 1) means inputs -1 and +1 should produce +1.
# The remaining examples cover the other two possible combinations.
# These target answers come from us; the network initially makes guesses.
# All four examples are training data; there is no separate test set here.
samples = [([-1, -1], -1), ([-1, 1], 1), ([1, -1], 1), ([1, 1], -1)]

# Repeat the indented code 200 times. range(200) supplies step values
# 0 through 199. step is a counter for the logs, not a network input.
# The colon starts a block; indentation shows which lines belong to it.
# Each iteration processes all four examples before one parameter update,
# so this is full-batch gradient descent.
for step in range(200):

    # Calculate the mean squared error using the current network parameters.
    # "for x, y in samples" unpacks each example into input list x and target y.
    # model(x) calls MLP.__call__ to compute one prediction as a Value object.
    # Subtract y to get the error, then **2 squares it. sum adds all four
    # squared errors; len(samples) is 4, so division gives their average.
    # The expression inside sum is a generator: it produces one term at a time.
    # loss is a Value containing both the numerical error (loss.data) and a
    # computation graph connecting that error to the weights and biases.
    # Each iteration builds a fresh graph; this line does not change weights.
    loss = sum((model(x) - y)**2 for x, y in samples) / len(samples); 

    # Clear every parameter's .grad field before calculating new derivatives.
    # The engine uses += to accumulate gradient contributions, so this prevents
    # previous training steps from contributing again. .data stays unchanged:
    # resetting gradients does not erase the weights the network has learned.
    # Python supplies model as self automatically when this method is called.
    model.zero_grad(); 

    # Backpropagate from the loss through its recorded computation graph.
    # This calculates each parameter's derivative of the average loss and
    # stores it in that parameter's .grad field. Contributions from all four
    # examples accumulate because the examples share the same parameters.
    # This line calculates derivatives; it does not update parameter values.
    loss.backward(); 

    # Print the current training step and loss to the terminal.
    # An f-string inserts expressions inside braces. {step} inserts the counter;
    # {loss.data:.6f} formats the numerical loss with six decimal places.
    # This only rounds the displayed text, not the value used for training.
    # The printed loss was calculated BEFORE the update on the next line.
    # There are 200 log messages; no extra loss is evaluated after the last update.
    print(f"Step {step}: loss={loss.data:.6f}")

    # Visit every trainable weight and bias, calling each actual Value object p.
    # The statement after the colon is the inner loop's one-line body.
    # p.data -= 0.05*p.grad means p.data = p.data - 0.05*p.grad:
    # subtract the gradient multiplied by the learning rate, 0.05.
    # A positive gradient moves the parameter down; a negative one moves it up.
    # Updating .data directly changes the network without recording the update
    # as another differentiable operation. All gradients are calculated before
    # any parameters are updated. The outer loop then starts the next step.
    for p in model.parameters(): p.data -= 0.05 * p.grad

# After training, collect the final numerical weights and biases into one list.
# Save it beside this script as weights.json, then print the same array.
import json
from pathlib import Path
weights = [p.data for p in model.parameters()]
Path(__file__).with_name("weights.json").write_text(json.dumps(weights), encoding="utf-8")
print("Trained weights and biases:", weights)


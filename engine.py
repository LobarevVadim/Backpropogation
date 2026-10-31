# WHY THIS CLASS EXISTS
# Python can already calculate 2 * 3. What ordinary numbers do not keep is
# an instruction saying "I came from multiplying these two particular inputs."
# Value is a number with a notebook: data holds the number, _prev remembers
# its inputs, and _backward remembers how to send derivatives to those inputs.
# Each weight, bias, and intermediate calculation can have its own notebook.
# This lets the same backward algorithm work for many different networks.
# A class is a blueprint; Value(2) creates one individual object from it.
# Indented code belongs to the class or function above it.
class Value:

    # This triple-quoted string is class documentation (a docstring),
    # not an instruction that changes the numerical calculation.
    """ stores a single scalar value and its gradient """

    # OBJECT INITIALIZATION: what happens when you write a = Value(2)?
    # Python creates an object and automatically passes it as self to __init__.
    # You supply 2; that becomes data. You do NOT supply self yourself.
    # Think: self means "this particular object that we are currently setting up."
    # After construction, a refers to that object. A different Value(2) would be
    # another object, with its own attributes, even though the numbers are equal.
    # The defaults _children=() and _op='' make those arguments optional:
    # Value(2) uses an empty tuple of inputs and an empty operation-label string.
    # An operation can instead call Value(5, (a, b), '+') to record its history.
    # The leading underscore means "intended for internal use", not enforced privacy.
    # __init__ initializes the object; it does not need to return the object.
    def __init__(self, data, _children=(), _op=''):

        # Store the supplied number on this object. A dot accesses an attribute.
        # For a = Value(2), this makes a.data equal to 2. self is the whole object;
        # data is just the number passed into this function.
        self.data = data

        # Create another attribute, initially zero. After loss.backward(), this
        # will hold d(loss)/d(this value): how sensitive the loss is to this number.
        self.grad = 0

        # Store a function for later. lambda: None is a tiny function taking no
        # arguments and doing nothing. A starting weight has no earlier inputs,
        # so it has nowhere to send gradients. Operation results replace this
        # placeholder with a function that knows the correct derivative rule.
        self._backward = lambda: None

        # Remember the input OBJECTS, not copies of their numerical data.
        # set makes a collection with no duplicates. For a*a, a appears once here,
        # but the multiplication derivative still counts both uses of a.
        # The name _children refers to the operation's inputs, looking backward.
        self._prev = set(_children)

        # Keep a text label such as '+' for inspecting the computation graph.
        # The math does not use this label to decide which derivative to calculate.
        self._op = _op

    # ADDITION: a + b asks Python to call a.__add__(b).
    # Here self is a and other is b. These double-underscore names are Python
    # hooks: the spelling connects the method to the + operator.
    # The method returns a NEW Value; it does not overwrite either input.
    def __add__(self, other):

        # Allow both a+b and a+3. isinstance asks whether other is a Value.
        # The expression "X if condition else Y" chooses one of two results.
        # Keep an existing Value, or wrap an ordinary number so it can join the graph.
        other = other if isinstance(other, Value) else Value(other)

        # Do the actual addition using ordinary Python numbers, then wrap its
        # result. For a=Value(2), b=Value(3), out.data is 5. The tuple (self, other)
        # records which objects produced it; '+' is just a descriptive label.
        out = Value(self.data + other.data, (self, other), '+')

        # Define a helper function now, but do not run it yet.
        # This inner function remembers self, other, and out from this operation.
        # That remembered connection is called a closure. Each operation gets its
        # own helper tied to its own inputs and output. out.grad will be filled in
        # later, during backpropagation from the final loss.
        def _backward():

            # For out = self + other, increasing self by a small amount increases
            # out by the same amount: the local derivative is 1. Multiply that by
            # out.grad using the chain rule. += ADDS to the existing gradient because
            # this input may affect the loss through several different branches.
            self.grad += out.grad

            # The other input also has local derivative 1. Add its contribution too.
            # If both inputs are the same object (a+a), these two lines add twice,
            # which correctly gives the derivative 2 before the upstream multiplier.
            other.grad += out.grad

        # Save the helper on the output so backward() can call it later.
        # _backward without parentheses refers to the function itself.
        # _backward() with parentheses would execute it immediately, which we do
        # not want: the final loss and its upstream gradients are not ready yet.
        out._backward = _backward

        # Give the newly created Value back to the caller. This allows chaining,
        # such as (a+b)*3: the returned Value can participate in the next operation.
        return out

    # MULTIPLICATION: a*b calls a.__mul__(b), with self=a and other=b.
    # The structure matches addition; the local derivative rules differ.
    def __mul__(self, other):

        # Allow both a+b and a+3. isinstance asks whether other is a Value.
        # The expression "X if condition else Y" chooses one of two results.
        # Keep an existing Value, or wrap an ordinary number so it can join the graph.
        other = other if isinstance(other, Value) else Value(other)

        # Multiply the ordinary numerical data and create a result remembering
        # both input objects. For a=2 and b=3, the new numerical result is 6.
        out = Value(self.data * other.data, (self, other), '*')

        # Define a helper function now, but do not run it yet.
        # This inner function remembers self, other, and out from this operation.
        # That remembered connection is called a closure. Each operation gets its
        # own helper tied to its own inputs and output. out.grad will be filled in
        # later, during backpropagation from the final loss.
        def _backward():

            # For z=a*b, dz/da=b. If b=3, moving a a little moves z three times
            # as much. Multiply this local slope by out.grad = d(loss)/dz.
            self.grad += other.data * out.grad

            # Likewise dz/db=a, so add a*out.grad to b's gradient.
            # For a*a, self and other refer to the same object; both contributions
            # are added, giving 2*a*out.grad as required.
            other.grad += self.data * out.grad

        # Save the helper on the output so backward() can call it later.
        # _backward without parentheses refers to the function itself.
        # _backward() with parentheses would execute it immediately, which we do
        # not want: the final loss and its upstream gradients are not ready yet.
        out._backward = _backward

        # Give the newly created Value back to the caller. This allows chaining,
        # such as (a+b)*3: the returned Value can participate in the next operation.
        return out

    # POWERS: a**2 calls a.__pow__(2). Here other is the exponent.
    # This implementation differentiates the base only; it does not support
    # a Value object as the exponent. ** is Python's exponentiation operator.
    def __pow__(self, other):

        # int means integer, float means a decimal floating-point number.
        # The tuple (int, float) lets isinstance accept either type. assert raises
        # an error with the supplied message if the check fails in normal execution.
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        # Compute the power. (self,) is a one-element tuple: its comma matters.
        # Only the base is a graph input here. An f-string substitutes expressions
        # inside braces, so f'**{other}' becomes '**2' when other is 2.
        out = Value(self.data**other, (self,), f'**{other}')

        # Define a helper function now, but do not run it yet.
        # This inner function remembers self, other, and out from this operation.
        # That remembered connection is called a closure. Each operation gets its
        # own helper tied to its own inputs and output. out.grad will be filled in
        # later, during backpropagation from the final loss.
        def _backward():

            # Apply the power rule: d(a**n)/da = n*a**(n-1).
            # For a=3 and n=2, the local derivative is 6. Multiply by the upstream
            # gradient and accumulate it, just as in addition and multiplication.
            self.grad += (other * self.data**(other-1)) * out.grad

        # Save the helper on the output so backward() can call it later.
        # _backward without parentheses refers to the function itself.
        # _backward() with parentheses would execute it immediately, which we do
        # not want: the final loss and its upstream gradients are not ready yet.
        out._backward = _backward

        # Give the newly created Value back to the caller. This allows chaining,
        # such as (a+b)*3: the returned Value can participate in the next operation.
        return out

    # RELU: call a.relu() explicitly; this is a normal method, not an
    # operator hook. It returns max(0, a) and remembers how to backpropagate.
    def relu(self):

        # The conditional expression chooses 0 for negative inputs and otherwise
        # keeps the input number. Record self as the single source of this output.
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        # Define a helper function now, but do not run it yet.
        # This inner function remembers self, other, and out from this operation.
        # That remembered connection is called a closure. Each operation gets its
        # own helper tied to its own inputs and output. out.grad will be filled in
        # later, during backpropagation from the final loss.
        def _backward():

            # The comparison produces True or False. In Python arithmetic these
            # act like 1 and 0. Positive outputs pass the upstream gradient through;
            # negative inputs get zero. At exactly zero, this implementation chooses
            # zero as the derivative because ReLU has no unique derivative there.
            self.grad += (out.data > 0) * out.grad

        # Save the helper on the output so backward() can call it later.
        # _backward without parentheses refers to the function itself.
        # _backward() with parentheses would execute it immediately, which we do
        # not want: the final loss and its upstream gradients are not ready yet.
        out._backward = _backward

        # Give the newly created Value back to the caller. This allows chaining,
        # such as (a+b)*3: the returned Value can participate in the next operation.
        return out

    # BACKPROPAGATION FOR THE WHOLE GRAPH: call loss.backward().
    # Here self is that loss object. Each _backward helper handles just one
    # operation; this public method arranges ALL helpers in the correct order.
    # We must collect every contribution to an intermediate node before using
    # its gradient to calculate contributions to earlier inputs.
    def backward(self):

        # Make an empty list for a topological ordering: inputs before outputs.
        # A list remembers order. We will later walk this list in reverse.
        topo = []

        # Make an empty set to track objects already explored. One weight may
        # be used in multiple places; its graph node should be scheduled only once.
        visited = set()

        # A recursive helper: it calls itself to explore earlier inputs.
        # v is simply the current Value being visited, not necessarily the loss.
        # The helper remembers topo and visited from its surrounding function.
        def build_topo(v):

            # Only explore this object if we have not seen it already.
            # The colon starts a block; indentation shows which statements belong to it.
            if v not in visited:

                # Mark this object as seen before exploring its inputs. add inserts an
                # item into a set. This avoids repeating work along shared graph paths.
                visited.add(v)

                # Loop through the direct input objects recorded when v was created.
                # For a leaf weight, this collection is empty, so the loop does nothing.
                for child in v._prev:

                    # Visit that input's inputs first. Each recursive call has its own v,
                    # but shares the same ordering list and visited set.
                    build_topo(child)

                # append places v at the end of the list AFTER all of its inputs.
                # For a -> b -> loss, this produces [a, b, loss].
                topo.append(v)

        # Start collecting the ordering from the output being differentiated.
        # Only nodes connected to this output will be included.
        build_topo(self)

        # Seed d(loss)/d(loss)=1. This starts the chain rule.
        # This assignment does NOT clear all other gradients. Reusing an old graph
        # can retain accumulated gradients; normal training builds a fresh forward
        # graph each step and clears parameter gradients with model.zero_grad().
        self.grad = 1

        # Walk backward through the ordering: loss first, then its inputs.
        # By the time we process a node, all later operations have contributed
        # to that node's gradient, including branches that meet at the same node.
        for v in reversed(topo):

            # Execute the derivative helper saved on this node. It adds contributions
            # to its inputs. Leaf nodes execute the do-nothing function from __init__.
            v._backward()

    # NEGATION: -a calls a.__neg__(). Reuse multiplication instead of writing
    # another derivative rule; multiplying by -1 already has the correct rule.
    def __neg__(self):

        # Return a new tracked multiplication result with the sign reversed.
        return self * -1

    # REFLECTED ADDITION: in 3+a, ordinary integer addition cannot handle a
    # Value, so Python can ask a.__radd__(3). The r means reflected: self is
    # the RIGHT operand in the original expression. Reuse our normal addition.
    def __radd__(self, other):

        # Swap the operands and let __add__ handle the calculation and graph.
        return self + other

    # SUBTRACTION: a-b calls a.__sub__(b). We already know how to add and
    # negate, so subtraction needs no new backward rule.
    def __sub__(self, other):

        # Calculate a + (-b); tracked operations build the necessary graph.
        return self + (-other)

    # REFLECTED SUBTRACTION: 3-a can call a.__rsub__(3).
    # Order matters: the answer must be 3-a, not a-3.
    def __rsub__(self, other):

        # Negate this Value, then add it to the original left operand.
        return other + (-self)

    # REFLECTED MULTIPLICATION: 3*a can call a.__rmul__(3).
    # Multiplication can swap its operands, so reuse __mul__.
    def __rmul__(self, other):

        # Return the tracked product using our existing multiplication method.
        return self * other

    # DIVISION: a/b calls a.__truediv__(b). A reciprocal is a power of -1,
    # so a/b = a*(b**-1). Reusing operations also reuses their derivative rules.
    def __truediv__(self, other):

        # Calculate the reciprocal first (** has higher precedence), then multiply.
        return self * other**-1

    # REFLECTED DIVISION: 3/a can call a.__rtruediv__(3).
    # Keep 3 as the numerator and this Value as the denominator.
    def __rtruediv__(self, other):

        # Multiply the original numerator by this Value raised to -1.
        return other * self**-1

    # READABLE REPRESENTATION: Python uses this method when displaying the
    # object, for example in an interactive terminal. It returns a string;
    # it does not print or add logs by itself.
    def __repr__(self):

        # An f-string inserts the current data and grad into a readable label,
        # for example Value(data=2, grad=3). This does not change the object.
        return f"Value(data={self.data}, grad={self.grad})"

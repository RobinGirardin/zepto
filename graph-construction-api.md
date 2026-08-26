The current graph construction API is a bit special in my opinion.
`Module` have their own `__call__` function, which then call the context `call_module()` method
to call the `Module.forward()`. The context method `call_method` function is to instantiate the module name and module path,
as well as call the module forward function. This part makes sense, somewhat, because the module path can only
be determine by the context.

In the `Module.forward`, the other present `Module` (sub-modules) have also their own `forward` function called, until
we achieve the last submodules that entirely composed of operations.

Such a module foward pass consists of using the context to `apply` each operation, hence adding the operation's node and its output tensors to the graph through `add_operation` and perform some validation. The `add_operation` necessits the operation input tensors to be provided inside the `apply()` method, and to be present already the graph as edges.
Following PyTorch style, tensors should be instantiated inside the `Module` constructor, alongside with other `Modules`.
The same should be the case for `Operation` if we were to follow  the PyTorch model declaration style. However,
this is not case: operation are declared directly through inside the `forward()` function inside their corresponding wrapper.

This present a contradiction.
Moreover, the fact that both `__call__` method and `forward` method necessits context make the tool call unintuitive:

`Module.__call__()` --> `context.call_module()` [Context usage] -> `Module.forward()` --> `context.apply(Operation(), inputs=(), parameters=())`

IMO, it would be much better to focus on the graph building through a single `__call__` function, for both `Module`, `Operation` and`Tensor`, while the `forward` method is reserved for `Module` to be composed of multiple `Operation` or `Tensor`.
This way, operation and tensors addition to the graph are deferred by the usage of their own `__call__` method.

This would also be the opportuniy of cleaning the whole tool call pipeline and API.

---

The current computational graph is composed of edges and node. Both edges and nodes are fixed and designed so that they're not mutable. They reason for this is to prevent any harmful changes in shape, thus bias the analysis.

* Edge: `Tensor`
* Node: `StructuralOperation`

Every edges are not new and independant entities throughout the graph.
Sometimes, an edge tensors is re-used elsewhere. As such,
every tensor has a storage id `Tensor.storage_id` which highlights shared tensors.

An edge binds to operation to together, one is the operation that produced the edge,
the other ones is an operation that consumes said edge.
To build a computation graph, the edge must track both of them.
`Tensor` does this through the `producer` and `consumer` attributes,
which hold reference to the producer or consumer operation(s).
By design choice, this reference is not the unique `Operation.id`.
The reason for this, is that this approach doesn't allow the graph to
know precisely which tensors goes to which operation ports.
The latter is important. A `MatMul` under the context of a linear projection operation
has two input port: `a` and `b`. One of them an activation tensors, while the other
is a model parameter. The distinction is important, because both have different shapes
and are focused by different cost estimation functionalities.
As such, to track which tensor refers to which output port, we use two addition intermediate classes.
`PortRef` and `PortSpec`. `PortRef` refers to an operation's ports. It holds
the operation id, to which the reference points to, and a port identifier `port_name`.
Each `Tensor.producer` and `Tensor.consumer` are tuple of `PortRef`.
Each operation's port are represented by as `PortSpec`, that contains a `name` – which `PortRef.port_name`
refers to – and information about the kind of tensors it recieves and transmits.

An edge refers to a tensor of a particular shape and with a particular functionality.
Those attributes of an edge are encoding through a `ValueMetadata` and then grephed to
the `Tensor` through its `metadata` attribute. This way, The `ValueMetadata` of a edge
can be freely created and modified, before being linked to a particular edge. Remember that
edges and node are designed to be immutable.

A graph is then composed of 5 main components:

* `Tensor`
* `StructuralOperation`
* `PortRef`
* `PortSpec`
* `ValueMetadata`

And then, the `StructuralGraph`, which contains the different edges and nodes.

A graph needs to be built. As said graph is immutable,
A high-level graph constructs it, before setting everything in stone and output
the final immutable `StructuralGraph`.
This class is called `StructuralGraphBuilder` and contains helper functions that
will add tensors, operations, and parameters, bind the edges and nodes together, until
the graph attain its final form. While `StructuralGraph` merely contains the final graph,
`StructuralGraphBuilder` merely builds the graph iteratively: it is told what to add where, and it executes.
It has no information about the any context, type of model of layer supplied, or overarching graph default, such as storage precision
for example. This is handled by the `GraphCompositionContext`, a window manager class
that instantiate a context object on starts, and deletes it as the end of the lifetime.
Within the context window, object can access the context through the `GraphCompositionContext.current()` method.

Under a specific context, provided by `GraphCompositionContext`, the user builds a model using
dedicated operations and submodules. This model is a class inheriting the `Module` parent class.
The model mathematical operation and tensors objects are first instantiated inside the the `__init__` function,
and then, executed inside a `forward()` method. Similarly to PyTorch model building API, the `__init__` function hold object of tensor or operation, holding their instruction on how to be used and how the computational should be built.
The actual graph build only happen the `forward` method, when all instantiating tensor or operation objects
are used to built the series of operation that the model contains.

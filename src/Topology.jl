module Topology

export  BinaryTreeTopology,
        NodeCoord,
        Path,
        nodes_in_layer,
        allnodes,
        leafnodes,
        getparent,
        getchildren,
        getneighbours,
        isleaf,
        distances_from_center,
        farthest_node,
        ttn_sweep_order

"""
    Topology

Module defining core data structures for representing hierarchical 
complete/perfect binary tree tensor network (TTN) topologies.

The focus is on **static, rooted, complete/perfect binary trees** suitable for
hierarchical TTN ansätze:

- The tree is organized into `L` layers, indexed from top to bottom:
  `ℓ = 1, 2, ..., L`.
- Within each layer `ℓ`, nodes are ordered from left to right and indexed:
  `pos = 1, 2, ..., n_ℓ`.
- A node is therefore naturally addressed by the tuple `(ℓ, pos)`.

This module does **not** deal with tensor data or physical indices.
It only encodes the *topological structure* of the tree.
"""


# ================================
# Type aliases
# ================================

"""
    NodeCoord = Tuple{Int,Int}

Type alias for a node coordinate in the binary tree.

A node is addressed by `(ℓ, pos)` where:

- `ℓ`   : layer index (1 = root layer, increases downward),
- `pos` : position within the layer, ordered from left to right.
"""
const NodeCoord = Tuple{Int,Int}


# ================================
# Core topology type
# ================================

"""
    BinaryTreeTopology

Topology of a hierarchical complete/perfect binary tree.

Fields
------

- `num_layers::Int`

    Total number of layers `L` in the tree.
    Layers are indexed from top (root) to bottom: `ℓ = 1, 2, ..., L`.

- `num_nodes_in_layer::Vector{Int}`

    Vector of length `num_layers` where `num_nodes_in_layer[ℓ]` is the
    number of nodes in layer `ℓ`, ordered from left to right.

Conventions
-----------

- `num_layers ≥ 1`;
- for all internal layers `ℓ < L`:

    ```julia
    num_nodes_in_layer[ℓ] == 2^ℓ
    ```

- for the bottom layer `ℓ = L`:

    ```julia
    1 ≤ num_nodes_in_layer[L] ≤ 2^L
    ```
"""
struct BinaryTreeTopology
    num_layers::Int
    num_nodes_in_layer::Vector{Int}

    """
        BinaryTreeTopology(num_nodes_in_layer::Vector{Int})

    Construct a `BinaryTreeTopology` from an explicit specification of the
    number of nodes in each layer:

    - `num_nodes_in_layer[ℓ]` is the number of nodes in layer `ℓ`,
      ordered from top (`ℓ = 1`) to bottom (`ℓ = L`).
    """
    function BinaryTreeTopology(num_nodes_in_layer::Vector{Int})
        num_layers = length(num_nodes_in_layer)
        num_layers >= 1 ||
            error("BinaryTreeTopology: a tree must contain at least one layer")

        # Check internal layers
        if num_layers > 1
            for ℓ in 1:num_layers-1
                expected = 2^ℓ
                actual = num_nodes_in_layer[ℓ]
                if actual != expected
                    error("BinaryTreeTopology: layer $ℓ must have $expected nodes, got $actual")
                end
            end
        end

        # Check bottom layer
        n_bottom = num_nodes_in_layer[end]
        max_nodes = 2^num_layers
        if n_bottom < 1 || n_bottom > max_nodes
            error("BinaryTreeTopology: bottom layer must have between 1 and $max_nodes nodes, got $n_bottom")
        end

        return new(num_layers, num_nodes_in_layer)
    end
end


"""
    BinaryTreeTopology(Nphys::Int)

Build a binary tree topology from the number of physical sites `Nphys`.

The computed `num_nodes_in_layer` is then passed to the inner constructor
`BinaryTreeTopology(num_nodes_in_layer::Vector{Int})`.
"""
function BinaryTreeTopology(Nphys::Int)
    @assert Nphys ≥ 4 "BinaryTreeTopology currently supports TTNs with Nphys ≥ 4."

    total_nodes = Nphys - 2

    # Top layer always has 2 nodes
    num_nodes_in_layer = Int[]
    push!(num_nodes_in_layer, 2)

    # Remaining nodes to distribute to lower layers
    remaining = total_nodes - 2  # we already placed 2 nodes in the top layer

    ℓ = 2 
    while remaining > 0
        # Maximum capacity of this layer: 2^ℓ nodes
        capacity = 2^ℓ

        # Put as many nodes as we can in this layer without exceeding `remaining`
        nodes_here = min(remaining, capacity)
        push!(num_nodes_in_layer, nodes_here)

        remaining -= nodes_here
        ℓ += 1
    end

    return BinaryTreeTopology(num_nodes_in_layer)
end


"""
    nodes_in_layer(topo::BinaryTreeTopology, ℓ::Int) -> Vector{NodeCoord}

Return a vector of all node coordinates `(ℓ, pos)` in layer `ℓ`,
ordered from left to right.

Throws an error if `ℓ` is not a valid layer index.
"""
function nodes_in_layer(topo::BinaryTreeTopology, ℓ::Int)::Vector{NodeCoord}
    if ℓ < 1 || ℓ > topo.num_layers
        error("nodes_in_layer: layer index ℓ = $ℓ is out of range 1:$(topo.num_layers)")
    end
    n = topo.num_nodes_in_layer[ℓ]
    
    return [(ℓ, pos) for pos in 1:n]
end


"""
    allnodes(topo::BinaryTreeTopology) -> Vector{NodeCoord}

Return a vector of all node coordinates of the tree `topo`,
ordered from the bottom layer up to the top layer.

Nodes inside each layer are listed left-to-right.
"""
function allnodes(topo::BinaryTreeTopology)::Vector{NodeCoord}
    nodes = NodeCoord[]
    for ℓ in topo.num_layers:-1:1
        append!(nodes, nodes_in_layer(topo, ℓ))
    end
    
    return nodes
end


"""
    leafnodes(topo::BinaryTreeTopology) -> Vector{NodeCoord}

Return all leaf nodes of the binary tree `topo`.

For complete binary trees, leaf nodes occur either in the bottom layer,
or in the second-lowest layer if the bottom layer is not full.

Nodes are returned left-to-right, starting with the bottom layer.
"""
function leafnodes(topo::BinaryTreeTopology)::Vector{NodeCoord}
    # Single-layer tree: everything is a leaf
    if topo.num_layers == 1
        return nodes_in_layer(topo, 1)
    end

    leaf_nodes = NodeCoord[]
    for ℓ in (topo.num_layers, topo.num_layers - 1)
        for node in nodes_in_layer(topo, ℓ)
            if isleaf(topo, node)
                push!(leaf_nodes, node)
            end
        end
    end

    return leaf_nodes
end


"""
    isperfect(topo::BinaryTreeTopology) -> Bool

Return `true` if the given `BinaryTreeTopology` represents a **perfect**
binary tree in the chosen convention.

A tree is perfect if its bottom layer is also full, i.e.

    num_nodes_in_layer[L] == 2^L

where `L = topo.num_layers`.
"""
function isperfect(topo::BinaryTreeTopology)::Bool
    L = topo.num_layers
    n_bottom = topo.num_nodes_in_layer[end]
    
    return n_bottom == 2^L
end


"""
    isvalidnode(topo::BinaryTreeTopology, node::NodeCoord) -> Bool

Return `true` if `node = (ℓ, pos)` is a valid node of the binary tree `topo`,
and `false` otherwise.

A node is valid if:
- `ℓ` is a valid layer index: 1 ≤ ℓ ≤ topo.num_layers
- `pos` is within the number of nodes in that layer: 1 ≤ pos ≤ num_nodes_in_layer[ℓ]
"""
function isvalidnode(topo::BinaryTreeTopology, node::NodeCoord)::Bool
    ℓ, pos = node

    # Check layer index
    if ℓ < 1 || ℓ > topo.num_layers
        return false
    end

    # Check position in that layer
    #n_in_layer = topo.num_nodes_in_layer[ℓ]
    return 1 ≤ pos ≤ topo.num_nodes_in_layer[ℓ] #n_in_layer
end


"""
    parent(topo::BinaryTreeTopology, node::NodeCoord) -> NodeCoord

Return the parent of `node = (ℓ, pos)` in the binary tree `topo`.

Conventions
-----------

- The top layer (`ℓ == 1`) is assumed to have exactly 2 nodes at positions
  `pos = 1, 2`. In this layer we define the parents *by convention* as

      parent((1,1)) = (1,2)
      parent((1,2)) = (1,1)

  so that every node in the network has exactly one parent.

- For all deeper layers (`ℓ ≥ 2`), the parent lives in layer `ℓ - 1` at
  position

      parent_pos = (pos + 1) ÷ 2

  which is the standard binary-tree rule.
"""
function getparent(topo::BinaryTreeTopology, node::NodeCoord)::NodeCoord
    # Validate address
    if !isvalidnode(topo, node) 
        error("parent: node $(node) is not a valid node")
    end
    
    ℓ, pos = node

    # Special convention for the top layer: 2 nodes that are each other's parent
    if ℓ == 1
        return pos == 1 ? (1,2) : (1,1)
    end

    # Standard parent rule for layers ℓ ≥ 2
    parent_layer = ℓ - 1
    parent_pos = (pos + 1) ÷ 2
    return (parent_layer, parent_pos)
end


"""
    getchildren(topo::BinaryTreeTopology, node::NodeCoord)
        -> Tuple{Union{NodeCoord,Nothing}, Union{NodeCoord,Nothing}}

Return the left and right children of `node = (ℓ, pos)` in the binary tree `topo`.

The result is a 2-tuple `(left_child, right_child)` where each entry is either:

- a valid `NodeCoord = (ℓ_child, pos_child)`, or
- `nothing` if that child does not exist.

Rules
-----
- If `node` is in the bottom layer (`ℓ == topo.num_layers`), return
  `(nothing, nothing)`.

- Otherwise, the potential children lie in layer `ℓ + 1` at:

      left_pos  = 2pos - 1
      right_pos = 2pos

  In a complete (non-perfect) tree, the bottom layer may be truncated.
  Only children with valid positions are returned; missing ones become `nothing`.
"""
function getchildren(topo::BinaryTreeTopology, node::NodeCoord)
    # Validate address
    if !isvalidnode(topo, node)
        error("getchildren: node $(node) is not a valid node")
    end

    ℓ, pos = node

    # Bottom layer: no children
    if ℓ == topo.num_layers
        return (nothing, nothing)
    end

    child_layer = ℓ + 1
    max_pos = topo.num_nodes_in_layer[child_layer]

    left_pos = 2 * pos - 1
    right_pos = 2 * pos

    left_child = left_pos <= max_pos ? (child_layer, left_pos) : nothing
    right_child = right_pos <= max_pos ? (child_layer, right_pos) : nothing

    return (left_child, right_child)
end


"""
    getneighbours(topo, node) -> Vector{NodeCoord}

Return all neighbouring nodes of `node` in the tree graph:
its parent and its children. Children that are `nothing`
are skipped.
"""
function getneighbours(topo::BinaryTreeTopology, node::NodeCoord)
    neighs = NodeCoord[]

    # First get parent 
    parent = getparent(topo, node)
    push!(neighs, parent)

    # Children: some may be `nothing` if missing
    for child in getchildren(topo, node)
        if child !== nothing
            push!(neighs, child)
        end
    end

    return neighs
end


"""
    is_leaf(topo::BinaryTreeTopology, node::NodeCoord) -> Bool

Return `true` if `node` is a leaf of the tree `topo`, i.e. it has no children.

Throws an error if `node` is not valid in `topo`.
"""
function isleaf(topo::BinaryTreeTopology, node::NodeCoord)::Bool
    if !isvalidnode(topo, node)
        error("is_leaf: node $(node) is not a valid node")
    end

    left_child, right_child = getchildren(topo, node)
    return left_child == nothing && right_child == nothing
end


"""
    Path

Represents a path between two nodes in a `BinaryTreeTopology`.

Fields
------

- `path_length::Int`
    number of steps one has to take to reach the end node (path_length = length(steps)-1).

- `anchor_coord::NodeCoord`
    The coordinate `(ℓ, pos)` of the anchor node.

- `anchor_pos::Int`
    The 1-based index of `anchor_coord` inside `steps`.

- `steps::Vector{NodeCoord}`
    Ordered list of coordinates forming the path from start to end.
"""
struct Path
    path_length::Int
    anchor_coord::NodeCoord
    anchor_pos::Int
    steps::Vector{NodeCoord}
end


"""
    Path(topo::BinaryTreeTopology, start_node::NodeCoord, end_node::NodeCoord) -> Path

Construct a `Path` object representing the simple path between
`start_node` and `end_node` in the tree `topo`.

Algorithm (high-level)
----------------------

1. Validate that both nodes belong to `topo`.
2. Handle two simple special cases:
   - `start_node == end_node` (degenerate path).
   - both nodes in the top layer (ℓ == 1).
3. Otherwise:
   - First while-loop: climb the deeper node up until both nodes are in
     the same layer, checking at each step if the climbing node meets
     the other node.
   - If they did not meet in that first phase, use a second while-loop:
     climb `start_node` and `end_node` alternately upward (one step each),
     checking after each step whether they coincide; this closes the path.
4. Combine the segments from the start-side and end-side climbs into
   a single `steps` vector, and determine `anchor_coord` and `anchor_pos`.

Special anchor convention
-------------------------

If the resulting path passes through the entire topmost layer, i.e. both
nodes `(1,1)` and `(1,2)` appear in `steps`, then:

- either could be considered an anchor;
- by convention, we choose `anchor_coord = (1,2)` and set
  `anchor_pos` to its index in `steps`.

Otherwise, the anchor is unique and determined by the climbing algorithm.
"""
function Path(topo::BinaryTreeTopology,
              start_node::NodeCoord,
              end_node::NodeCoord)::Path

    # ------------------------------
    # 0. Validation
    # ------------------------------
    if !isvalidnode(topo, start_node)
        error("Path: start_node $(start_node) is not a valid node")
    end
    if !isvalidnode(topo, end_node)
        error("Path: end_node $(end_node) is not a valid node")
    end

    # ------------------------------
    # 1. Trivial special cases
    # ------------------------------

    # Case 1: same node
    if start_node == end_node
        steps = [start_node]
        anchor_coord = start_node
        anchor_pos = 1
        path_length = length(steps) - 1
        return Path(path_length, anchor_coord, anchor_pos, steps)
    end

    ℓ_start, _ = start_node
    ℓ_end, _ = end_node

    # Case 2: both nodes in the top layer
    if ℓ_start == 1 && ℓ_end == 1
        steps = [start_node, end_node]

        # By convention, if the path crosses the top layer,
        # the anchor is (1,2).
        anchor_coord = (1,2)
        anchor_pos = findfirst(n -> n == (1,2), steps)

        path_length = length(steps) - 1
        return Path(path_length, anchor_coord, anchor_pos, steps)
    end

    # ------------------------------
    # 2. Setup for general case
    # ------------------------------

    # Chains from each endpoint up towards the anchor
    start_chain = NodeCoord[start_node]
    end_chain   = NodeCoord[end_node]

    cur_start = start_node
    cur_end = end_node
    ℓ_start = cur_start[1]
    ℓ_end = cur_end[1]

    # Helper: if needed, enforce anchor = (1,2) when path crosses full top layer
    adjust_anchor = steps -> begin
        pos12 = findfirst(n -> n == (1,2), steps)
        return ((1,2), pos12)
    end

    # Variables that will hold the final result before common finalization
    steps = NodeCoord[]
    anchor_coord = start_node  # temporary init, will be overwritten
    anchor_pos = 1
    found_direct = false       # did we close the path in the first loop?

    # ------------------------------
    # 3. First while-loop:
    #    climb the deeper node to the same layer
    # ------------------------------

    # Case: start_node is deeper
    if ℓ_start > ℓ_end
        while ℓ_start > ℓ_end
            parent_start = getparent(topo, cur_start)
            push!(start_chain, parent_start)

            # If we climb directly onto end_node, path closes here
            if parent_start == cur_end
                steps = start_chain
                anchor_coord = cur_end
                anchor_pos = length(steps)
                found_direct = true
                break
            end

            cur_start = parent_start
            ℓ_start = cur_start[1]
        end
    # Case: end_node is deeper
    elseif ℓ_end > ℓ_start
        while ℓ_end > ℓ_start
            parent_end = getparent(topo, cur_end)
            push!(end_chain, parent_end)

            # If we climb directly onto start_node, path closes here
            if parent_end == cur_start
                # end_chain: [end_node, ..., start_node]
                steps = reverse(end_chain)  # start_node → ... → end_node
                anchor_coord = cur_start
                anchor_pos = 1
                found_direct = true
                break
            end

            cur_end = parent_end
            ℓ_end = cur_end[1]
        end
    end

    # ------------------------------
    # 4. If first phase did not close the path,
    #    use second while-loop: climb both nodes
    # ------------------------------

    if !found_direct
        # Now cur_start and cur_end are in the same layer but not equal.
        anchor_coord_any = nothing

        while true
            # Step 1: climb from start side
            parent_start = getparent(topo, cur_start)
            push!(start_chain, parent_start)
            cur_start = parent_start

            if cur_start == cur_end
                anchor_coord_any = cur_start
                break
            end

            # Step 2: climb from end side
            parent_end = getparent(topo, cur_end)
            push!(end_chain, parent_end)
            cur_end = parent_end

            if cur_end == cur_start
                anchor_coord_any = cur_end
                break
            end
        end

        # Build steps from chains:
        # start_chain: [start_node → ... → anchor]
        # end_chain:   [end_node   → ... → anchor]
        while !isempty(end_chain) && end_chain[end] == anchor_coord_any
            pop!(end_chain)
        end

        steps = NodeCoord[]
        append!(steps, start_chain)          # start_node → ... → anchor
        append!(steps, reverse(end_chain))   # anchor → ... → end_node

        anchor_coord = anchor_coord_any::NodeCoord
        anchor_pos   = length(start_chain)
    end

    # ------------------------------
    # 5. Common finalization
    # ------------------------------

    # Apply top-layer anchor convention only if needed
    if anchor_coord != (1,2) && any(n -> n == (1,1), steps) && any(n -> n == (1,2), steps)
        anchor_coord, anchor_pos = adjust_anchor(steps)
    end

    path_length = length(steps) - 1
    return Path(path_length, anchor_coord, anchor_pos, steps)
end


"""
    distances_from_center(topo::BinaryTreeTopology, center_coord::NodeCoord)
        -> Dict{Int, Vector{Tuple{NodeCoord,NodeCoord}}}

Compute the distance structure of the tree `topo` relative to the node
`center_coord`.

Returns a dictionary `distances` such that:

- the keys are graph distances `d = 1, 2, 3, ...` (number of edges)
  from `center_coord`,
- the values are vectors of 2-tuple node coordinates :
    - the first node of the tuple is a node at distance d from the center node
    - the second node is the next node on the path from the first node to the center node (useful for TTN initialization)

    distances[d] = [ nodes (together with the next nodes on the path) whose shortest path to center_coord has length d ]
"""
function distances_from_center(topo::BinaryTreeTopology, center_coord::NodeCoord)::Dict{Int, Vector{Tuple{NodeCoord,NodeCoord}}}
    if !isvalidnode(topo, center_coord)
        error("distances_from_center: center node $(center_coord) is not a valid node")
    end

    leaf_nodes = leafnodes(topo)
    distances = Dict{Int, Vector{Tuple{NodeCoord,NodeCoord}}}()
    visited = Set{NodeCoord}()

    for leaf in leaf_nodes
        # Skip center if it appears as a leaf
        if leaf == center_coord
            continue
        end

        path = Path(topo, leaf, center_coord)
        dist = path.path_length   # start with distance leaf → center

        for (i, cur_node) in enumerate(path.steps)
            # Skip the center explicitly
            if cur_node == center_coord
                break
            end

            # If already processed, just decrement distance and continue
            if cur_node ∈ visited
                dist -= 1
                continue
            end

            # First time encountering this node
            push!(visited, cur_node)
            bucket = get!(distances, dist, Tuple{NodeCoord,NodeCoord}[])
            push!(bucket, (cur_node, path.steps[i+1]))

            dist -= 1
        end
    end

    return distances
end


"""
    farthest_node(topo, node) -> NodeCoord

Return the node that is maximally far (in graph distance) from `node`.

If multiple nodes have the same maximal distance, choose the one with
larger `pos` coordinate; if that still ties, choose the one in the
upper layer (smaller layer index).
"""
function farthest_node(topo::BinaryTreeTopology, node::NodeCoord)
    distances = distances_from_center(topo, node)
    # Maximal distance
    max_d = maximum(keys(distances))

    # Extract candidate nodes at maximal distance
    level = distances[max_d]  # vector of 2-tuples
    candidates = NodeCoord[ pair[1] for pair in level ]

    # Tie-breaker: larger pos; if tie, smaller layer index (upper layer)
    best = candidates[1]
    for n in candidates
        if n[2] > best[2] || (n[2] == best[2] && n[1] < best[1])
            best = n
        end
    end

    return best
end


"""
    ttn_sweep_order(topo) -> Vector{NodeCoord}

Return an ordered list of all TTN nodes specifying the update
order in a single *forward* sweep, using a leaf-to-leaf depth-first
strategy.

The start leaf is chosen as the leftmost leaf in the deepest layer,
and the end leaf as the rightmost leaf in the deepest layer. The
backward sweep can be done by reversing this order.
"""
function ttn_sweep_order(topo::BinaryTreeTopology)
    # 1. Start: leftmost node in the bottom layer
    n_layers  = length(topo.num_nodes_in_layer)
    start_node = (n_layers, 1)

    # 2. End: farthest from start_node, using distances_from_center
    end_node = farthest_node(topo, start_node)

    # 3. Main path from start to end, using your Path type
    p = Path(topo, start_node, end_node)
    main_path = p.steps  # Vector{NodeCoord}
    
    # 3. DFS sweep: explore side branches off the main path
    visited = Set{NodeCoord}()
    order = NodeCoord[]

    # Recursive DFS for side branches
    function dfs_branch(node::NodeCoord)
        if node in visited
            return
        end
        push!(visited, node)
        push!(order, node)
        for nb in getneighbours(topo, node)
            dfs_branch(nb)
        end
    end

    # Walk along main_path, exploring side branches at each node
    for (k, node) in enumerate(main_path)
        # Visit node itself if not yet visited
        if !(node in visited)
            push!(visited, node)
            push!(order, node)
        end

        # Neighbours on main path directly before/after this node
        prev_on_path = k > 1                 ? main_path[k-1] : nothing
        next_on_path = k < length(main_path) ? main_path[k+1] : nothing

        # Explore all neighbours that are not prev/next on main path
        for nb in getneighbours(topo, node)
            if nb != prev_on_path && nb != next_on_path && !(nb in visited)
                dfs_branch(nb)
            end
        end
    end

    return order
end


end # module Topology
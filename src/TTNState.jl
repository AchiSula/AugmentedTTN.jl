module TTNState

using ITensors, ITensorMPS
using Base.Threads: @threads, nthreads
using ThreadsX

using ..Topology  
using ..Operators
import ..Operators: TTNLinkOps

export  BinaryTTN, 
        gettensor, 
        settensor!, 
        shift_center!,
        single_site_update!


"""
    fuse_indices(child1::Index{Int}, child2::Index{Int}, max_bond_dim::Int) -> Index{Int}

Given two child indices `child1` and `child2`, return a new "fused" index
whose dimension is `min(dim(child1) * dim(child2), max_bond_dim)`.
"""
function fuse_indices(child1::Index{Int}, child2::Index{Int}, max_bond_dim::Int)::Index{Int}
    fused_dim = min(dim(child1) * dim(child2), max_bond_dim)

    # Temporary tag; will be replaced by `tag_virtual_link`.
    return Index(fused_dim, "Link")
end


"""
    tag_virtual_link(link::Index,
                     child_node::NodeCoord,
                     parent_node::NodeCoord) -> Index

Return a new Index with the same dimension as `link` but with a
descriptive tag encoding the TTN edge it belongs to.

The tag has the form:

    "Link,(ℓ_child,pos_child),(ℓ_parent,pos_parent)"

with the convention that the *child* coordinate is always listed first.

Special case: for the top horizontal link between (1,1) and (1,2),
the tag is always

    "Link,(1,1),(1,2)"

regardless of which one is passed as `child_node` or `parent_node`.
"""
function tag_virtual_link(link::Index,
                          child_node::NodeCoord,
                          parent_node::NodeCoord)::Index
    # Enforce special ordering for the top horizontal link
    is_top_pair = (child_node == (1,1) && parent_node == (1,2)) ||
                  (child_node == (1,2) && parent_node == (1,1))

    if is_top_pair
        # Convention: always (1,1) first, then (1,2)
        tag_str = "Link,p1.1,p1.2"
    else
        child_for_tag = child_node
        parent_for_tag = parent_node
        tag_str = "Link,c$(child_for_tag[1]).$(child_for_tag[2]),p$(parent_for_tag[1]).$(parent_for_tag[2])"
    end

    return Index(dim(link), tag_str)
end


"""
    generate_indices(topo::BinaryTreeTopology,
                     phys_inds::Vector{Index{Int}},
                     max_bond_dim::Int) -> Vector{Vector{ITensor}}

Construct the *index layout* (skeleton) of a dense Binary TTN specified by
`topo`, using the physical site indices `phys_inds` (ordered left-to-right),
and the maximum virtual bond dimension `max_bond_dim`.

Returns a nested vector `tensors[ℓ][i]`, where each entry is an *empty*
3-leg `ITensor` containing only index structure:

    inds(tensors[ℓ][i]) == (parent_ind, left_child_ind, right_child_ind)

No tensor data is initialized; the output is suitable for subsequent
random initialization and isometrization passes.

"""
function generate_indices(topo::BinaryTreeTopology,
                          phys_inds::Vector{Index{Int}},
                          max_bond_dim::Int)

    num_layers = topo.num_layers
    Nphys = length(phys_inds)

    # Container for all ITensors of TNN (index-only skeleton)
    tensors = Vector{Vector{ITensor}}(undef, num_layers)

    # ============================================================
    # Case 1: ONLY ONE LAYER
    # ============================================================
    if num_layers == 1
        top_nodes = nodes_in_layer(topo, 1)
        @assert length(top_nodes) == 2 "Single-layer TTN must have exactly two nodes in layer 1."

        @assert Nphys == 4 "Single-layer TTN currently assumes exactly 4 physical indices."

        nodeL = top_nodes[1]  # (1,1)
        nodeR = top_nodes[2]  # (1,2)

        # Children of nodeL
        left_child_ind_1 = phys_inds[1]
        right_child_ind_1 = phys_inds[2]

        # Children of nodeR
        left_child_ind_2 = phys_inds[3]
        right_child_ind_2 = phys_inds[4]

        # Top horizontal link dimension from both halves
        x = dim(left_child_ind_1) * dim(right_child_ind_1)
        y = dim(left_child_ind_2) * dim(right_child_ind_2)
        dim_top = min(x, y, max_bond_dim)

        raw_top_link = Index(dim_top, "Link")
        top_link = tag_virtual_link(raw_top_link, nodeL, nodeR)

        tensors[1] = ITensor[
            ITensor(top_link, left_child_ind_1, right_child_ind_1),
            ITensor(top_link, left_child_ind_2, right_child_ind_2),
        ]

        return tensors
    end

    # ============================================================
    # Case 2: TWO OR MORE LAYERS
    # ============================================================

    # ------------------------------------------------------------
    # 1. Bottom layer (ℓ = num_layers)
    # ------------------------------------------------------------

    bottom_nodes = nodes_in_layer(topo, num_layers)
    B = length(bottom_nodes)

    @assert Nphys ≥ 2B "Not enough physical indices for bottom layer: need at least $(2B), got $Nphys."

    tensors[num_layers] = Vector{ITensor}(undef, B)

    # Counter for consumed physical indices (1-based)
    phys_pos = 1

    for (i, node) in enumerate(bottom_nodes)
        # Bottom nodes always attach two physical children: (2i-1, 2i)
        left_child_ind  = phys_inds[phys_pos]
        right_child_ind = phys_inds[phys_pos + 1]
        phys_pos += 2

        raw_parent_ind = fuse_indices(left_child_ind, right_child_ind, max_bond_dim)
        parent_node = getparent(topo, node)
        parent_ind = tag_virtual_link(raw_parent_ind, node, parent_node)

        # ITensor(parent, left_child, right_child)
        tensors[num_layers][i] = ITensor(parent_ind, left_child_ind, right_child_ind)
    end

    # Check if tree is structurally perfect
    perfect = Topology.isperfect(topo)

    # ------------------------------------------------------------
    # 2. Second-lowest layer (ℓ = num_layers - 1)
    # ------------------------------------------------------------

    ℓ = num_layers - 1
    layer_nodes = nodes_in_layer(topo, ℓ)
    NL = length(layer_nodes)
    tensors[ℓ] = Vector{ITensor}(undef, NL)

    if perfect
        # Perfect tree: second-lowest layer nodes are all internal (two virtual children)
        @assert phys_pos == Nphys + 1 "Perfect tree: all physical indices should be used in bottom layer."

        for (i, node) in enumerate(layer_nodes)
            # Children live in bottom layer at positions 2i-1 and 2i
            left_child_ind  = inds(tensors[num_layers][2i - 1])[1]
            right_child_ind = inds(tensors[num_layers][2i])[1]

            raw_parent_ind = fuse_indices(left_child_ind, right_child_ind, max_bond_dim)
            parent_node = getparent(topo, node)
            parent_ind  = tag_virtual_link(raw_parent_ind, node, parent_node)

            tensors[ℓ][i] = ITensor(parent_ind, left_child_ind, right_child_ind)
        end
    else
        # Complete tree: internal, hybrid, and leaf nodes in this layer
        internal_count = B ÷ 2
        has_hybrid = (B % 2 == 1)
        hybrid_pos = internal_count + 1

        for (i, node) in enumerate(layer_nodes)

            if i ≤ internal_count
                # Internal node: two virtual children from bottom layer
                left_child_ind  = inds(tensors[num_layers][2i - 1])[1]
                right_child_ind = inds(tensors[num_layers][2i])[1]

            elseif has_hybrid && i == hybrid_pos
                # Hybrid node: left virtual, right physical
                left_child_ind = inds(tensors[num_layers][2i - 1])[1]

                @assert phys_pos ≤ Nphys "Not enough physical indices for hybrid node."
                right_child_ind = phys_inds[phys_pos]
                phys_pos += 1

            else
                # Leaf node in this layer: two physical children
                @assert phys_pos + 1 ≤ Nphys "Not enough physical indices for leaf node."
                left_child_ind  = phys_inds[phys_pos]
                right_child_ind = phys_inds[phys_pos + 1]
                phys_pos += 2
            end

            raw_parent_ind = fuse_indices(left_child_ind, right_child_ind, max_bond_dim)
            parent_node = getparent(topo, node)
            parent_ind = tag_virtual_link(raw_parent_ind, node, parent_node)

            tensors[ℓ][i] = ITensor(parent_ind, left_child_ind, right_child_ind)
        end

        @assert phys_pos == Nphys + 1 "Complete tree: not all physical indices were consumed."
    end

    # ------------------------------------------------------------
    # 3. Remaining upper layers (ℓ = num_layers-2 down to 1)
    #    including the top layer
    # ------------------------------------------------------------

    for ℓ in num_layers-2:-1:1
        layer_nodes = nodes_in_layer(topo, ℓ)
        NL = length(layer_nodes)
        tensors[ℓ] = Vector{ITensor}(undef, NL)

        for (i, node) in enumerate(layer_nodes)
            # Children live in layer ℓ+1
            left_child_node, right_child_node = getchildren(topo, node)

            # For these layers, both children are TTN nodes (internal layers)
            left_child_pos  = left_child_node[2]
            right_child_pos = right_child_node[2]

            left_child_ind  = inds(tensors[ℓ + 1][left_child_pos])[1]
            right_child_ind = inds(tensors[ℓ + 1][right_child_pos])[1]

            raw_parent_ind = fuse_indices(left_child_ind, right_child_ind, max_bond_dim)
            parent_node = getparent(topo, node)
            parent_ind  = tag_virtual_link(raw_parent_ind, node, parent_node)

            tensors[ℓ][i] = ITensor(parent_ind, left_child_ind, right_child_ind)
        end
    end

    # ------------------------------------------------------------
    # 4. Final unification of the top horizontal link
    #    (nodes (1,1) and (1,2) must share the SAME Index)
    # ------------------------------------------------------------

    top_nodes = nodes_in_layer(topo, 1)
    @assert length(top_nodes) == 2 "Top layer must contain exactly two nodes."

    # Parent indices (horizontal link) of each root tensor
    parentL_ind = inds(tensors[1][1])[1]
    parentR_ind = inds(tensors[1][2])[1]

    # We want both roots to use the *same* Index.
    # Keep the one with smaller dimension and replace the other.
    if dim(parentL_ind) ≤ dim(parentR_ind)
        # Keep parentL_ind; replace parentR_ind with parentL_ind
        tensors[1][2] = ITensor(parentL_ind, inds(tensors[1][2])[2], inds(tensors[1][2])[3])
    else
        # Keep parentR_ind; replace parentL_ind with parentR_ind
        tensors[1][1] = ITensor(parentR_ind, inds(tensors[1][1])[2], inds(tensors[1][1])[3])
    end

    return tensors
end


"""
    renormalize_link_ops!(tpo::TTNLinkOps, T::ITensor, inward_ind::Index, outward_inds::Vector{<:Index})

Renormalize local TPO operator pieces from the two outward links of a TTN
node onto its inward link during TTN initialization or DMRG sweeps.

Steps performed for each term `i`:

  • Retrieve TPO pieces `op_1`, `op_2` acting on `outward_inds[1]` and
    `outward_inds[2]`.  
  • If both pieces are absent, skip this term.  
  • If only one outward operator is present, the other leg is traced out by
    replacing the corresponding primed index on `conj(prime(T))`.  
  • Contract: `op_inward = T * op_1/op_2 * conj(prime(T))`.  
  • If `op_inward` has exactly two indices, accumulate all such two-leg
    operators into a single combined operator on the inward link.
    Otherwise, store `op_inward` under term id `i`.

Multi-threading
---------------
This function is thread-safe.  
Each call writes only to the `LinkOps` associated with the **unique**
`inward_ind` for that node, so parallel execution over different nodes
never mutates the same dictionary. Outward-link data is read-only.

Arguments
---------
- `tpo::TTNLinkOps`  
    The global operator container where renormalized TPO pieces are stored.
- `T::ITensor`  
    The isometry tensor on the current node (already QR-orthogonalized).
- `inward_ind::Index`  
    The index representing the link pointing toward the TTN center.
- `outward_inds::Vector{<:Index}`  
    The two indices of the outward links of this node.

Returns
-------
`nothing`
"""
function renormalize_link_ops!(
    tpo::TTNLinkOps, 
    T::ITensor,
    inward_ind::Index,
    outward_inds::Vector{<:Index}
    )

    if !haskey(tpo.TTN_link_ops, tags(inward_ind))
        error("renormalize_link_ops!: tag $(tags(inward_ind)) is missing from TTNLinkOps")
    end

    # Important: reset LinkOps on inward_ind before recomputing
    tpo.TTN_link_ops[tags(inward_ind)] = LinkOps()

    # Get LinkOps.link_ops dicts on each outward link
    link_ops_1 = tpo.TTN_link_ops[tags(outward_inds[1])].link_ops
    link_ops_2 = tpo.TTN_link_ops[tags(outward_inds[2])].link_ops

    # Create and prime conjugate T
    T_dag_primed = conj(prime(T))

    # For those terms i that consist of only one piece (a tpo piece with no internal links); records the first occurance
    first_i_two_legs = 0
    
    for i in 1:tpo.n_ops
        # Get the operator pieces acting on outward links corresponding to term i    
        op_1 = get(link_ops_1, i, nothing)
        op_2 = get(link_ops_2, i, nothing)

        if op_1 === nothing && op_2 === nothing
            continue
        end
        
        T_dag = T_dag_primed

        if op_1 === nothing && op_2 !== nothing
            outward_ind_1 = outward_inds[1]
            T_dag = replaceind(T_dag, prime(outward_ind_1), outward_ind_1) # replace primed outward_ind_1 on T_dag with outward_ind_1
            op_inward = T * op_2 * T_dag
        elseif op_1 !== nothing && op_2 === nothing
            outward_ind_2 = outward_inds[2]
            T_dag = replaceind(T_dag, prime(outward_ind_2), outward_ind_2) # replace primed outward_ind_2 on T_dag with outward_ind_2
            op_inward = T * op_1 * T_dag
        elseif op_1 !== nothing && op_2 !== nothing
            op_inward = T * op_1 * op_2 * T_dag
        end

        # Update tpo (sum all two leg tpo pieces into one piece)
        if length(inds(op_inward)) == 2
            if first_i_two_legs == 0
                first_i_two_legs = i
                tpo.TTN_link_ops[tags(inward_ind)].link_ops[first_i_two_legs] = op_inward
            else
                tpo.TTN_link_ops[tags(inward_ind)].link_ops[first_i_two_legs] += op_inward
            end
        else
            tpo.TTN_link_ops[tags(inward_ind)].link_ops[i] = op_inward
        end
    end # for loop

    return
end


"""
    initialize_random_TTN!(tensors, topo, center_coord=(1,2); eltype=ComplexF64)

Random initialization of a binary TTN described by `tensors` and `topo`,
putting the network into a unitary/central gauge with respect to `center_coord`.

After initialization, all non-center nodes are isometries pointing toward `center_coord`,
and the center tensor is a normalized random tensor of type `eltype`.
"""
function initialize_random_TTN!(
    tensors::Vector{Vector{ITensor}},
    topo::BinaryTreeTopology,
    center_coord::NodeCoord = (1,2);
    htpo::TTNLinkOps,
    eltype::Type = ComplexF64
)
    
    distances = distances_from_center(topo, center_coord) #  Dict{Int, Vector{Tuple{NodeCoord,NodeCoord}}}
    max_distance = maximum(keys(distances))

    # Start with nodes maximally furthest from the center and gradually move inwards 
    for dist in max_distance:-1:1
        pairs = distances[dist]  # Vector{Tuple{NodeCoord,NodeCoord}}
        n = length(pairs)
        if n == 0
            continue
        end

        # For each edge (node → next_node), store the old and new shared Indices (inward_ind, new_qr_ind)
        link_info = Vector{Tuple{Index,Index}}(undef, n)

        # Phase 1: parallel work on each node at distance d
        @threads for idx in 1:n
            node, next_node = pairs[idx]

            T = tensors[node[1]][node[2]]
            T_next = tensors[next_node[1]][next_node[2]]

            inward_ind = commonind(T, T_next) # the index of T that leads to the center_node
            outward_inds = uniqueinds(inds(T), inward_ind) # the other indices of T that lead away from the center_node

            # Create a random isometry
            T_rand = random_itensor(eltype, inds(T))
            Q, R = qr(T_rand, outward_inds; positive=true, tags=tags(inward_ind))
            qr_ind = commonind(Q, R)  # new inward link index

            tensors[node[1]][node[2]] = Q
            link_info[idx] = (inward_ind, qr_ind)

            # Renormalize inward_ind if htpo contains operators
            if htpo.n_ops > 0
                renormalize_link_ops!(htpo, Q, qr_ind, outward_inds)
            end
        end

        # Phase 2: serial updates of each next_node (only replace shared index)
        for (idx, (node, next_node)) in enumerate(pairs)
            inward_ind, qr_ind = link_info[idx]

            T_next = tensors[next_node[1]][next_node[2]]
            T_next_inds = collect(inds(T_next))

            # Replace old inward_ind with qr_ind
            for (i, ind) in enumerate(T_next_inds)
                if tags(ind) == tags(inward_ind)
                    T_next_inds[i] = qr_ind
                    break
                end
            end

            tensors[next_node[1]][next_node[2]] = ITensor(T_next_inds)
        end
    end

    # Center tensor: random + normalize 
    center_T = tensors[center_coord[1]][center_coord[2]]
    center_T = random_itensor(eltype, inds(center_T))
    center_T ./= norm(center_T)
    tensors[center_coord[1]][center_coord[2]] = center_T

    return
end


"""
    BinaryTTN

Container for a binary TTN state defined on a fixed `BinaryTreeTopology`.

Fields
------

- `topo::BinaryTreeTopology`

    The tree topology (structure of nodes and layers).

- `tensors::Vector{Vector{ITensor}}`

    A nested vector of ITensors organized by layers:

        tensors[ℓ][pos]  is the tensor at node (ℓ, pos).
"""
mutable struct BinaryTTN
    topo::BinaryTreeTopology
    max_bond_dim::Int
    center_coord::NodeCoord
    tensors::Vector{Vector{ITensor}}
    phys_inds::Vector{<:Index}
    htpo::TTNLinkOps # Hamiltonian TPO on TTN links

    """
    BinaryTTN(phys_inds::Vector{<:Index}, max_bond_dim::Int;
              center_coord::NodeCoord = (1,2),
              eltype::Type = ComplexF64,
              opsum::OpSum = OpSum())

    Construct a randomly initialized, isometric binary TTN state.
    
    Arguments
    ---------
    
    - `phys_inds::Vector{<:Index}`
    
        Physical site indices ordered from left to right. The number of
        physical indices must satisfy `Nphys ≥ 4`. Smaller systems are
        currently not supported by this TTN constructor.
    
    - `max_bond_dim::Int`
    
        Maximum bond dimension used when generating virtual link indices
        in `generate_indices`.
    
    Keyword arguments
    -----------------
    
    - `center_coord::NodeCoord = (1,2)`
    
        The coordinate of the isometry center.
        All other nodes are isometries pointing toward this center.
    
    - `eltype::Type = ComplexF64`
    
        Element type of the random initial tensors.
    
    - `opsum::OpSum = OpSum()`
    
        Placeholder for a future Hamiltonian
    """
    function BinaryTTN(
        phys_inds::Vector{<:Index},
        max_bond_dim::Int;
        center_coord::NodeCoord = (1,2),
        eltype::Type = ComplexF64,
        opsum::OpSum = OpSum(),
    )
        Nphys = length(phys_inds)
        @assert Nphys ≥ 4 "BinaryTTN currently supports TTNs with Nphys ≥ 4 sites."

        # 1. Build tree topology from Nphys
        topo = BinaryTreeTopology(Nphys)

        # 2. Sanity check: center_coord must be a valid node
        if !Topology.isvalidnode(topo, center_coord) 
            error("BinaryTTN: center node $(center_coord) is not a valid node")
        end

        # 3. Generate ITensor containers with correct index layout
        tensors = generate_indices(topo, phys_inds, max_bond_dim)

        # 4. If opsum is not empty, build Hamiltonian TPO container (on physical links only for now) 
        htpo = TTNLinkOps(tensors, phys_inds; opsum=opsum, eltype=eltype)
        #end

        # 5. Randomly initialize and isometrize around center_coord
        initialize_random_TTN!(tensors, topo, center_coord; eltype=eltype, htpo=htpo)

        return new(topo, max_bond_dim, center_coord, tensors, phys_inds, htpo)
    end
end


"""
    gettensor(TTN::BinaryTTN, node::NodeCoord) -> ITensor

Return the ITensor stored at node `(ℓ, pos)` in the TTN.
"""
function gettensor(TTN::BinaryTTN, node::NodeCoord)
    ℓ, pos = node
    if !Topology.isvalidnode(TTN.topo, node)
        error("gettensor: invalid node $(node) for the TTN topology.")
    end
    
    return TTN.tensors[ℓ][pos]
end


"""
    settensor!(TTN::BinaryTTN, node::NodeCoord, T::ITensor) -> BinaryTTN

Replace the ITensor at node `(ℓ, pos)` with `T`.  
Returns the modified TTN.
"""
function settensor!(TTN::BinaryTTN, node::NodeCoord, T::ITensor)
    ℓ, pos = node
    if !Topology.isvalidnode(TTN.topo, node)
        error("settensor!: invalid node $(node) for the TTN topology.")
    end
    
    TTN.tensors[ℓ][pos] = T
    
    return TTN
end


"""
    TTNLinkOps(TTN::BinaryTTN; opsum=OpSum(), eltype::Type = ComplexF64)

High-level constructor for building operator tpo structures directly from a TTN.
"""
function TTNLinkOps(TTN::BinaryTTN; opsum::OpSum=OpSum(), eltype::Type = ComplexF64)
    TTNLinkOps(TTN.tensors, TTN.phys_inds; opsum=opsum, eltype=eltype)
end


"""
    shift_center!(TTN::BinaryTTN, old_center::NodeCoord, new_center::NodeCoord)

Shift the isometry center of the TTN from `old_center` to `new_center` in-place.

Assumptions:
  * `TTN` is already in canonical/isometric form with respect to `old_center`
    (e.g. freshly initialized by `initialize_random_TTN!` or the result of a
    previous `shift_center!` call).
  * `old_center` and `new_center` are valid nodes of `TTN.topo`.

This preserves the TTN state up to gauge and produces a new canonical form
with center at `new_center`.
"""
function shift_center!(
    TTN::BinaryTTN, 
    old_center::NodeCoord, 
    new_center::NodeCoord
    )

    # Sanity checks
    if old_center == new_center
        error("shift_center!: TTN is already isometrised toward $(old_center)!")
    end

    if TTN.center_coord != old_center
        error("shift_center!: provided old_center $old_center does not match TTN.center_coord = $(TTN.center_coord).")
    end

    if !Topology.isvalidnode(TTN.topo, new_center)
        error("shift_center!: new_center $new_center is not a valid node.")
    end
        
    path = Path(TTN.topo, old_center, new_center)
    for (idx, cur_node) in enumerate(path.steps)
        if cur_node == new_center
            break
        end

        # Current and next tensors
        next_node = path.steps[idx+1]
        T_cur = gettensor(TTN, cur_node)
        T_next = gettensor(TTN, next_node)

        # Inward vs outward indices
        inward_ind = commonind(T_cur, T_next)
        outward_inds = uniqueinds(inds(T_cur), inward_ind)

        # QR-decompose T_cur to make it isometry
        Q, R = qr(T_cur, outward_inds; positive=true, tags=tags(inward_ind))
        inward_ind = commonind(Q, R)

        # Update TTN
        settensor!(TTN, cur_node, Q)
        settensor!(TTN, next_node, R * T_next)

        # Renormalize updated inward_ind if htpo contains operators
        if TTN.htpo.n_ops > 0
            renormalize_link_ops!(TTN.htpo, Q, inward_ind, outward_inds)
        end
    end 

    # Update iso center
    TTN.center_coord = new_center 
    
    return nothing
end


"""
    linearmapi(vec_in, link_ops_vec, i) -> ITensor

Apply a single Hamiltonian term `i` of the TTN effective Hamiltonian
to the center tensor `vec_in`.

For a given term `i`, this function multiplies `vec_in` by all link
operators that exist for that term, one per link, and returns the
resulting ITensor.
"""
function linearmapi(vec_in::ITensor, link_ops_vec::Vector{Dict{Int, ITensor}}, i::Int)    
    vec_i_out = vec_in
    for idx in 1:length(link_ops_vec)
        link_op_i = get(link_ops_vec[idx], i, nothing)
        if link_op_i !== nothing
            vec_i_out *= link_op_i
        end
    end

    prime!(vec_i_out, -1; plev=1)
    
    return vec_i_out
end


"""
    linearmap(vec_in, link_ops_vec, env_i) -> ITensor

Apply the full effective Hamiltonian `H_eff` to the current center tensor `vec_in`.

`link_ops_vec` is a vector whose elements are dictionaries mapping
term indices `i::Int` to the corresponding link operator ITensors on
each virtual link attached to the center node.

This function computes

    H_eff * vec_in = sum_{i ∈ env_i} H_i * vec_in

in a matrix-free way, using `ThreadsX.mapreduce` to parallelize over
the term index `i`.
"""
function linearmap(vec_in::ITensor, link_ops_vec::Vector{Dict{Int, ITensor}}, env_i::Vector{Int})
    return ThreadsX.mapreduce(i -> linearmapi(vec_in, link_ops_vec, i), +, env_i)
end  


"""
    single_site_update!(TTN, center_coord; kwargs...) -> Real

Perform a single-site TTN-DMRG update at the node `center_coord`.

Treating the TTN as being in a unitary gauge with isometry
center at `center_coord`, this function:

1. Builds the effective Hamiltonian `H_eff` acting on the center tensor
   from the renormalized link operators stored in `TTN.htpo`.
2. Uses `ITensors.eigsolve` (KrylovKit-based Lanczos) to find the
   lowest-energy eigenvector of `H_eff`, starting from the current
   center tensor as an initial guess.
3. Replaces the center tensor in `TTN` with this optimized eigenvector.

Keyword arguments (with defaults matching KrylovKit):
- `krylovdim::Int = 30`: dimension of the Krylov subspace.
- `which::Symbol = :SR`: which eigenvalue to target (`:SR` = smallest real).
- `tol::Float64 = 1e-12`: convergence tolerance for the eigensolver.
- `maxiter::Int = 100`: maximum number of Lanczos iterations.

Returns the lowest eigenvalue at this center (local energy).
Throws an error if `TTN.htpo.n_ops == 0`, i.e. if no Hamiltonian TPO
was provided.
"""
function single_site_update!(
    TTN::BinaryTTN, 
    center_coord::NodeCoord; 
    krylovdim::Int = 30, 
    which::Symbol = :SR, 
    tol::Float64 = 1E-12, 
    maxiter::Int = 100
    )  

    if TTN.htpo.n_ops == 0
        error("single_site_update!: Hamiltonian TPO does not exist!")
        return 
    end
    
    center_tensor = gettensor(TTN, center_coord)
    center_inds = inds(center_tensor)

    # env_i contains all non-trivial terms i for the current center
    env_i_set = Set{Int}()
    # Holds the references to LinkOps on the center node links
    link_ops_vec = Vector{Dict{Int, ITensor}}()

    for ind in center_inds
        # Dict{Int, ITensor} for this link
        link_ops = TTN.htpo.TTN_link_ops[tags(ind)].link_ops
        push!(link_ops_vec, link_ops)

        # Accumulate all i that appear on any link
        for i in keys(link_ops)
            push!(env_i_set, i)
        end
    end 
    env_i = collect(env_i_set) 
    
    linmap(tensor_in::ITensor) = linearmap(tensor_in::ITensor, link_ops_vec, env_i)
    
    # Eigen decompose
    eg, ev_ITensor = ITensorMPS.eigsolve(linmap, center_tensor, 1, which; maxiter=maxiter, tol=tol, krylovdim=krylovdim, ishermitian=true);
    
    # Update TTN 
    settensor!(TTN, center_coord, ev_ITensor[1])
    
    return eg[1]
end


end # module TTNState
module Operators

using ITensors, ITensorMPS

export  LinkOps,
        TTNLinkOps

"""
    LinkOps

Container holding all local operator pieces (TPO pieces) acting on a
specific TTN link (physical or virtual).

Fields
------
- `link_ops::Dict{Int, ITensor}` :
    Maps each global term id `i` (e.g., the i-th OpSum term) to the
    corresponding operator ITensor acting on this link.

Notes
-----
This struct stores only the local operator parts. Complete TTN operators
(e.g. Hamiltonians or observables) are represented by combining the
`LinkOps` objects across all links inside a `TTNLinkOps`.
"""
struct LinkOps
    link_ops::Dict{Int, ITensor}   # term_id -> local operator ITensor

    # trivial inner constructor: initialize empty ops dictionary
    LinkOps() = new(Dict{Int, ITensor}())
end


"""
    process_ops!(TTN_link_ops::Dict{TagSet, LinkOps}, opsum::OpSum, phys_inds::Vector{<:Index}, eltype::Type) -> Int

Process a user–provided `OpSum` into a TPO (tensor–product operator)
pieces acting on the **physical links** of a TTN.

This function:

  • Iterates over all raw operator terms in `opsum`  
  • Extracts coefficients and operator identities  
  • Builds a ITensor TPO (via `MPO`) for each term  
  • Retags the internal link indices with `"i=<term_id>"`  
  • Inserts each resulting ITensor TPO pieces into the appropriate
    `LinkOps` object of the `TTN_link_ops` dictionary, keyed by the
    index tag of the corresponding TTN physical index 

LinkOps on virtual links remain empty and will be populated later during TTN renormalization procedure.

Returns
-------
`Int`  
    The total number of operator terms processed (i.e. `length(opsum)`).

Throws
------
`Error`  
    If `opsum` is empty or if a coefficient cannot be converted to the
    requested `eltype`.
"""
function process_ops!(TTN_link_ops::Dict{TagSet, LinkOps}, opsum::OpSum, phys_inds::Vector{<:Index}, eltype::Type)::Int
    if length(opsum) < 1
        error("TTNLinkOps: OpSum holds no operators!")
    end

    n_ops = length(opsum)
    
    # Create ITensor TPOs for each term i and insert them in the appropriate LinkOps objects 
    for i in 1:n_ops
        op_raw = opsum[i]
        co = coefficient(op_raw) # extract coefficient
        co = try
            convert(eltype, co)
        catch e
            if e isa InexactError || e isa MethodError
                error("Cannot convert coefficient $co of type $(typeof(co)) to element type $eltype.")
            else
                rethrow(e)
            end
        end
        
        # Term i acts on multiple sites
        if length(op_raw) > 1
            tpo_inds = Index[] # to collect all indices (Index objects) on which the TPO opsum[i] acts
            tpo_tags = String[] # to collect all tags (specifying the identities of TPO pieces) of the TPO opsum[i] 

            # Loop through TPO pieces and store their indentities, the sites they act on (integer label + phys_ind)
            for idx in 1:length(op_raw)
                site_piece = op_raw[idx].sites[1]
                tag_piece = op_raw[idx].which_op
                push!(tpo_inds, phys_inds[site_piece])
                push!(tpo_tags, tag_piece)
            end
            
            # Create ITensor TPO representing term i
            tpo_i = co * MPO(eltype, tpo_inds, tpo_tags)

            # Modify the tags of the internal indices of tpo_i to specify which term they correspond to
            for idx in 1:(length(op_raw) - 1)
                internal_inds = linkinds(tpo_i, idx)
                for internal_ind in internal_inds
                    internal_ind_tagged = addtags(internal_ind, "i=$(i)")
                    replaceind!(tpo_i[idx], internal_ind, internal_ind_tagged)
                    replaceind!(tpo_i[idx+1], internal_ind, internal_ind_tagged)
                end
            end

            # Fill LinkOps on physical indices
            for idx in 1:length(tpo_i)
                tpo_piece = tpo_i[idx]
                TTN_link_ops[tags(tpo_inds[idx])].link_ops[i] = tpo_piece
            end
        # Term i acts on only one site
        else
            site = op_raw[1].sites[1]
            tag = op_raw[1].which_op
            tpo_i = co * MPO(eltype, [phys_inds[site]], [tag])

            # Fill LinkOps
            TTN_link_ops[tags(phys_inds[site])].link_ops[i] = tpo_i[1]
        end
    end

    return n_ops
end


"""
    TTNLinkOps(tensors::Vector{Vector{ITensor}}, phys_inds::Vector{<:Index}; opsum=OpSum(), eltype=ComplexF64)

Container holding `LinkOps` objects for **all** TTN links.

Fields
------
- `TTN_link_ops::Dict{TagSet, LinkOps}` :
    For each TTN link (identified by its index TagSet), stores a `LinkOps`
    object containing all local operator pieces acting on that link.

- `n_ops::Int` :
    Total number of global operator terms (e.g., total number of OpSum
    terms). Initially 0 until filled by `process_ops!`.

Constructor
-----------
Builds an empty operator structure skeleton for a given TTN index layout.
For every index appearing in the TTN.tensors (implicitly specifies TTN index layout), 
its TagSet is extracted, and an empty `LinkOps` container is created for it.

The constructor guarantees that the operator container has one entry for each
physical and virtual link of the TTN.

Additionally, if OpSum object is provided, the constructor will process it to fill LinkOps
with operator tensors (TPO format) on the physical legs.
"""
struct TTNLinkOps
    TTN_link_ops::Dict{TagSet, LinkOps}   # map from link tags → LinkOps
    n_ops::Int                            # total number of operator terms in a given opsum

    # Inner constructor: build skeleton from TTN tensors
    function TTNLinkOps(tensors::Vector{Vector{ITensor}}, phys_inds::Vector{<:Index}; opsum::OpSum = OpSum(), eltype::Type = ComplexF64)
        tags_seen = Set{TagSet}()

        # Collect all index tags from the TTN
        for layer in tensors
            for T in layer
                for i in inds(T)
                    push!(tags_seen, tags(i))
                end
            end
        end

        # Create empty LinkOps for each link tag
        TTN_link_ops = Dict{TagSet, LinkOps}()
        for t in tags_seen
            TTN_link_ops[t] = LinkOps()
        end

        if length(opsum) == 0
            return new(TTN_link_ops, 0)
        else
            n_ops = process_ops!(TTN_link_ops, opsum, phys_inds, eltype)
            return new(TTN_link_ops, n_ops)
        end
    end
end

end # module Operators
module DMRG

using ITensors#, ITensorMPS
using JLD2
using Printf

using ..Topology  
using ..Operators
using ..TTNState

export dmrg


"""
    dmrg(TTN_ini::BinaryTTN, nsweeps::Int=3; kwargs...) -> energy, TTN

Run a single-site DMRG optimization on a `BinaryTTN` state.

This routine performs a sequence of sweeps through the TTN, optimizing 
one tensor at a time using `single_site_update!`. After each 
update, the center is moved to the next node in the sweep, according 
to the predefined sweep order.

# Arguments
- `TTN_ini::BinaryTTN`  
    Initial Tree Tensor Network to optimize. A deep copy is made 
    internally so the input is not modified.

- `nsweeps::Int`  
    Number of DMRG sweeps to perform.

# Keyword Arguments
- `krylovdim::Int = 30`  
    Dimension of the Krylov subspace used by `eigsolve`.

- `which::Symbol = :SR`  
    Eigenvalue selection rule passed to `eigsolve` (e.g. `:SR`, `:LR`).

- `eigsolve_tol::Float64 = 1e-12`  
    Tolerance passed to the eigensolver.

- `eigsolve_maxiter::Int = 100`  
    Maximum number of Lanczos/Arnoldi iterations.

- `eps_abs::Float64 = 1e-14`  
    Absolute energy convergence threshold. If the energy change between 
    consecutive sweeps is below this value, DMRG stops early.

- `nsweeps_prev_runs::Int = 0`  
    Optional offset added to printed sweep numbers (useful when 
    continuing a previous run).

- `save_TTN::AbstractString = ""`  
    If nonempty, save the TTN after each sweep to JLD2 format at the 
    given path.

# Returns
- `energy::Float64`  
    Final energy estimate after the last sweep.

- `TTN::BinaryTTN`  
    The optimized TTN after DMRG convergence or completion of all sweeps.

# Notes
This is a basic single-site DMRG implementation for now.

"""
function dmrg(TTN_ini::BinaryTTN, nsweeps::Int=3; kwargs...)  
    # eigsolve kwargs
    krylovdim = get(kwargs, :krylovdim, 30)::Int
    which = get(kwargs, :which, :SR)::Symbol 
    eigsolve_tol = get(kwargs, :eigsolve_tol, 1E-12)::Float64  # ::Union{Float64, Array{Float64, 1}}
    eigsolve_maxiter = get(kwargs, :eigsolve_maxiter, 100)::Int

    # Energy convergence control
    eps_abs = get(kwargs, :eps_abs, 1E-14)::Float64

    # Total number of sweeps in all the previous runs (might be useful later; for now it's always 0)
    nsweeps_prev_runs = get(kwargs, :nsweeps_prev_runs, 0)::Int

    # Save TTN after each sweep in jld2 format, if the path/name (string) is provided.
    save_TTN = get(kwargs, :save_TTN, "")::AbstractString

    # Create sweep order: both forward and backawrd
    forward_order = ttn_sweep_order(TTN_ini.topo)          
    backward_order = reverse(forward_order[1:end-1])             
    sweep_order = vcat(forward_order, backward_order)
    sweep_length = length(sweep_order)

    TTN = deepcopy(TTN_ini)

    # Shift (if needed) the iso center toward the first node in the sweep order
    if sweep_order[1] != TTN.center_coord
        shift_center!(TTN, TTN.center_coord, sweep_order[1])
    end
    
    # Energy expectation value
    energy = 0.0  
    previous_energy = 0.0

    # All sweeps
    for sw in 1:nsweeps
        previous_energy = energy
        maxtruncerr = 0.0 # used later for subspace expansion
        max_bond_dim = TTN.max_bond_dim
        
      sw_time = @elapsed begin
        # Sweep through TTN and update every node
        for idx in 1:sweep_length                
            # Next center in the sweep order
            idx != sweep_length ? next_center_coord = sweep_order[idx+1] : next_center_coord = sweep_order[1]    

            # Update current center
            energy = single_site_update!(TTN, TTN.center_coord; krylovdim=krylovdim, which=which, tol=eigsolve_tol, maxiter=eigsolve_maxiter)

            # Shift iso center
            if TTN.center_coord != next_center_coord
                shift_center!(TTN, TTN.center_coord, next_center_coord)
            end  
        end      
      end # end elapsed
        
        #TTN.max_bond_dim = max_bond_dim (used later)
        
        @printf(
        "After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
          sw + nsweeps_prev_runs,
          energy,
          max_bond_dim,
          maxtruncerr,
          sw_time
        )
        flush(stdout)

        # Save the TTN at the end of sweep (sw + nsweeps_prev_runs) if needed
        if !isempty(save_TTN)
            save(save_TTN, "TTN", TTN, "sweep", sw + nsweeps_prev_runs, "energy", energy)
        end
    
        if sw > 1 && abs(previous_energy - energy) < eps_abs 
            @printf(
              "Desired energy convergence achieved after sweep %d\n",
              sw + nsweeps_prev_runs
            )
            flush(stdout)
            
            return energy, TTN
        end    
    end # end sweeps

    return energy, TTN
end


end # module DMRG
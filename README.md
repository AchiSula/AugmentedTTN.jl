# AugmentedTTN

A Julia package implementing (augmented) Tree Tensor Networks (TTNs), built on top of ITensors.jl [ITensors.jl Documentation](https://docs.itensor.org/ITensors/stable/).

## Overview

Tree Tensor Networks (TTNs) are tensor-network ansätze widely used for representing quantum many-body states and for performing 
numerical simulations of strongly correlated systems. Their defining feature is a loop-free, hierarchical structure, which enables
efficient tensor contractions, well-defined canonical forms, and stable variational optimization algorithms.

However, standard TTNs are most effective in one-dimensional systems and face fundamental limitations in higher dimensions. 
In particular, conventional TTN architectures do not satisfy the area law for entanglement entropy in more than one dimension, which 
restricts their ability to accurately represent ground states of genuinely two- or higher-dimensional quantum systems.

Augmented Tree Tensor Networks address this limitation by adding an additional augmentation layer to the underlying TTN structure, 
allowing the network to encode the area law. The augmentation mechanism implemented in this project is inspired by the approach 
described in [https://arxiv.org/abs/2011.08200](https://arxiv.org/abs/2011.08200).

Additionally, the package implements both bosonic and fermionic variants of (augmented) TTNs. Incorporating fermionic degrees of freedom 
into tensor-network algorithms introduces an additional layer of complexity due to sign flips associated with fermionic exchange statistics.
This requires careful handling of fermion parity and operator oredering. The fermionization strategy adopted here is based on approaches 
originally developed  for fermionic PEPS architectures, in particular the formalism introduced in [arXiv:0912.0646](https://arxiv.org/abs/0912.0646).

## DMRG for Tree Tensor Networks

The primary optimization algorithm implemented in this package is the Density Matrix Renormalization Group (DMRG) [arXiv:cond-mat/0409292](https://arxiv.org/abs/cond-mat/0409292). It is a variational optimization algorithm that exploits the loop-free structure of TTNs 
to optimize the network one tensor at a time. By fixing all tensors except a single local "center" tensor,  the global optimization problem 
reduces to a sequence of local variational problems that can be solved efficiently.

In this formulation, the objective (cost) function is the ground-state energy of the quantum system, evaluated as a full contraction of the
Tree Tensor Network with the Hamiltonian operators and its conjugate Tree.The network tensors are typically initialized randomly. 
By sweeping through the network and visiting each tensor one by one, successive local updates progressively lower the energy. Due to 
the variational nature of the algorithm, the energy is monotonically reduced toward the true ground-state energy from above.

Local tensor updates, effective Hamiltonian construction, environment contractions, and the handling of symmetric tensors are directly based
on the formalism presented in [The Tensor Networks Anthology:Simulation techniques for many-body quantum lattice systems](https://arxiv.org/abs/1710.03733), 
which provides an extensive and unified overview of loop-free tensor network architectures, including Tree Tensor Networks.

## Example Code: DMRG with MPS vs TTN

The following example demonstrates how to construct a randomly initialized TTN and optimize it using DMRG, and compares the workflow
to an MPS-based DMRG calculation using ITensors. Both approaches target the ground state of a one-dimensional spin-1/2 Heisenberg chain.

```julia
using AugmentedTTN
using ITensors, ITensorMPS

let
N = 16
sites = siteinds("S=1/2", N)
max_bond_dim = 25

opsum = OpSum()
for j=1:N-1
    opsum += 0.5,"S+",j,"S-",j+1
    opsum += 0.5,"S-",j,"S+",j+1
    opsum += "Sz",j,"Sz",j+1
end
    
# Create a Hamiltonian MPO for MPS 
H = MPO(opsum, sites)

nsweeps = 10
maxdim = [max_bond_dim] 
cutoff = [1E-12] # desired truncation error

# MPS (ITensor)
psi0 = random_mps(sites; linkdims=2)
println("MPS")
energy_mps, psi = ITensorMPS.dmrg(H, psi0; nsweeps, maxdim, cutoff)

# TTN 
TTN_ini = BinaryTTN(sites, max_bond_dim; center_coord=(1,2), opsum=opsum, eltype=ComplexF64)
println("\nTTN")
energy_TTN, TTN = AugmentedTTN.dmrg(TTN_ini, nsweeps)
end
```

The output below shows the convergence of the ground-state energy for both MPS- and TTN-based DMRG calculations using the same bond dimension:
```text
MPS
After sweep 1 energy=-6.874345609787575  maxlinkdim=8 maxerr=0.00E+00 time=14.591
After sweep 2 energy=-6.911721002539687  maxlinkdim=25 maxerr=6.07E-10 time=0.049
After sweep 3 energy=-6.911737131097764  maxlinkdim=25 maxerr=1.16E-09 time=0.120
After sweep 4 energy=-6.911737131389156  maxlinkdim=25 maxerr=8.07E-10 time=0.083
After sweep 5 energy=-6.9117371314201534 maxlinkdim=25 maxerr=8.05E-10 time=0.108
After sweep 6 energy=-6.911737131429781  maxlinkdim=25 maxerr=8.04E-10 time=0.081
After sweep 7 energy=-6.911737131438129  maxlinkdim=25 maxerr=8.04E-10 time=0.096
After sweep 8 energy=-6.911737131444997  maxlinkdim=25 maxerr=8.04E-10 time=0.089
After sweep 9 energy=-6.911737131450151  maxlinkdim=25 maxerr=8.04E-10 time=0.127
After sweep 10 energy=-6.911737131453782 maxlinkdim=25 maxerr=8.04E-10 time=0.099

TTN
After sweep 1 energy=-6.911737141326616  maxlinkdim=25 maxerr=0.00E+00 time=3.482
After sweep 2 energy=-6.91173714246288   maxlinkdim=25 maxerr=0.00E+00 time=0.382
After sweep 3 energy=-6.911737142517188  maxlinkdim=25 maxerr=0.00E+00 time=0.293
After sweep 4 energy=-6.911737142527842  maxlinkdim=25 maxerr=0.00E+00 time=0.312
After sweep 5 energy=-6.9117371425345215 maxlinkdim=25 maxerr=0.00E+00 time=0.288
After sweep 6 energy=-6.9117371425397645 maxlinkdim=25 maxerr=0.00E+00 time=0.303
After sweep 7 energy=-6.911737142544022  maxlinkdim=25 maxerr=0.00E+00 time=0.327
After sweep 8 energy=-6.911737142547525  maxlinkdim=25 maxerr=0.00E+00 time=0.335
After sweep 9 energy=-6.911737142550446  maxlinkdim=25 maxerr=0.00E+00 time=0.421
After sweep 10 energy=-6.911737142552886 maxlinkdim=25 maxerr=0.00E+00 time=0.290
```

In the example above, the initial setup—site indices, Hamiltonian definition via `OpSum`, and the construction of the Hamiltonian 
Matrix Product Operator (MPO)—follows exactly the same workflow as in a standard ITensor-based calculation 
(see the ITensor documentation: [https://docs.itensor.org/Overview/](https://docs.itensor.org/Overview/)).

In contrast to MPS-based DMRG, which operates on the MPO level, the Tree Tensor Network implementation takes the raw `OpSum` 
describing the Hamiltonian. From this input, the TTN code constructs and maintains renormalized effective operators acting on the 
virtual links of TTN during the DMRG optimization.



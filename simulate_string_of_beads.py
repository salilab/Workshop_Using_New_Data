# File:     custom_string_of_beads.py
# Synopsis: Brownian Dynamics simulation of a parametrized strings of beads.
#           The radius of gyration of the chain is reported, to demonstrate
#           how external information can be integrated by means of either
#           representation choices (string of beads, harmonic restraints
#           between consecutive beads) and by filtering (choosing parameter
#           combinations that reproduce experimental measurements).
# Author:   Barak Raveh
# E-mail:   barak at salilab dot org
# Date last
# changed:  Dec 16, 2016
# Note:     This simplified model of a string of beads is designed mainly
#           for illustrative purposes, but it could be easily extended
#           to a more realistic model that accounts for attractive
#           interactions, etc.

import IMP
import IMP.algebra
import IMP.atom
import IMP.container
import IMP.core
import IMP.rmf
import RMF
import math
import sys

# ==
# Params:
# ==
# force coefficient for harmonic restraint between bonded chain particles in kcal/mol/A2:
if len(sys.argv)>1:
    bond_k= float(sys.argv[1])
else:
    bond_k= 0.2
# scaling of resting length between a pair of bonded particle centers, relative to their sum of radii:
if len(sys.argv)>2:
    rest_length_factor= float(sys.argv[2])
else:
    rest_length_factor= 1.0
if len(sys.argv)>3:
    simulation_time_factor= float(sys.argv[3])
else:
    simulation_time_factor= 1.0
print("Bond k: %.3f" % bond_k)
print("Bond rest length factor: %.2f" % rest_length_factor)
print("Simulation time factor: %.3f" % simulation_time_factor)
radius= 6 # radius of coarse grained particles in A
bond_rest_length= 2*radius*rest_length_factor # rest length between consective chain beads in A
excluded_k= 10.0 # force coefficient for excluded volume restrain between non-bonded chain particles
                 # in kcal/mol/A2
nres= 260 # number of residues in a chain
nres_per_particle= 20 # number of residues represented by a single particle
box_side_A= 2000 # simulation box size
N= int(round(nres/nres_per_particle)) # number of particles in coarse-grained chain
n_inner_cycles= 10000 # number of frames in each inner simulation cycles
n_outer_cycles= int(math.ceil(250*simulation_time_factor)) # number of outer simulation cycles (total = n_inner_cycles x n_outer_cycles)
time_step_fs= 100.0 # Simulating time step per inner simulation cycle (frame)
slack_A= 10.0 # slack parameter for close pair container (affects performance only, not results)
DISABLE_RMF=True
rmf_fname= "string_of_beads.rmf" # name for output RMF file
rmf_output_interval_frames= 500 # interval for RMF output in frames (multiple by time_step_fs for time interval)


def create_diffusing_particle(m, radius):
    '''
    create an IMP particle that can be used as valid input for an IMP
    BrownianDynamics simulation, having coordinates, radius, mass and
    diffusion coefficient
    '''
    p= IMP.Particle(m)
    d= IMP.core.XYZR.setup_particle(p)
    d.set_radius(radius)
    d.set_coordinates_are_optimized(True)
    m= IMP.atom.Mass.setup_particle(p, 1)
    diff= IMP.atom.Diffusion.setup_particle(p)
    return p

# ==========
# == Main ==
# ==========

# ==
# I. Create chain particles for simulation
# ==
m=IMP.Model()
P=[]
bb=IMP.algebra.BoundingBox3D([0,0,0],[box_side_A,box_side_A,box_side_A])
chain_root_p= IMP.Particle(m)
chain_root_h= IMP.atom.Hierarchy.setup_particle( chain_root_p )
for i in range(N):
    p= create_diffusing_particle(m, radius)
    c= IMP.algebra.get_random_vector_in(bb)
    IMP.core.XYZ(p).set_coordinates(c)
    P.append(p)
    h= IMP.atom.Hierarchy.setup_particle( p )
    chain_root_h.add_child( h )



# ==
# II. Create scoring fucntion with chain bonds and excluded volume restraints
#     on chain particles
# ==
# bonded:
bond_score= IMP.core.HarmonicDistancePairScore(bond_rest_length,
                                               bond_k,
                                               "bond score")
bonded_pairs= IMP.container.ExclusiveConsecutivePairContainer(m, P) # return all consecutive pairs in P
bond_restraint= IMP.container.PairsRestraint(bond_score, bonded_pairs, "bond restraint")
# excluded volme:
excluded_score= IMP.core.SoftSpherePairScore(excluded_k)
close_pairs= IMP.container.ClosePairContainer(P, 0, slack_A) # container that returns overlapping pairs of particles
close_pairs.add_pair_filter( IMP.container.ExclusiveConsecutivePairFilter() ) # filter out bonded pairs
excluded_restraint= IMP.container.PairsRestraint(excluded_score, close_pairs, "excluded volume restraint")
# final scoring function:
scoring_function= IMP.core.RestraintsScoringFunction \
    ([bond_restraint, excluded_restraint])

# ==
# III. Create BrownianDynamics simulation
# ==
bd=IMP.atom.BrownianDynamics(m)
bd.set_maximum_time_step(time_step_fs)
bd.set_scoring_function(scoring_function)

# ==
# IV. Equilibrate system
# ==
print("Equilibrating system")
old_score= 100000 # large number
while old_score -scoring_function.evaluate(False)>0.01:
    bd.optimize(n_inner_cycles*50)
bd.optimize(n_inner_cycles*100)


# ==
# V. Create and link output RMF file to model
# ==
if(not DISABLE_RMF):
    rmf= RMF.create_rmf_file(rmf_fname)
    IMP.rmf.add_hierarchy(rmf, chain_root_h)
    IMP.rmf.add_restraint(rmf, bond_restraint)
    IMP.rmf.add_restraint(rmf, excluded_restraint)
    os = IMP.rmf.SaveOptimizerState(m, rmf)
    os.set_log_level(IMP.SILENT)
    os.set_period(rmf_output_interval_frames) # output interval
    bd.add_optimizer_state(os)

# ==
# VI. Simulate and track radius of gyration statistics
# ==
ERg2=0.0
ERg4=0.0
print("Simulating for %.0f ns" \
          % (n_outer_cycles*n_inner_cycles*time_step_fs/1E6))
for i in range(n_outer_cycles):
    bd.optimize(n_inner_cycles)
    Rg= IMP.atom.get_radius_of_gyration(P)
    ERg2= ERg2 + (Rg**2)/n_outer_cycles
    ERg4= ERg4 + (Rg**4)/n_outer_cycles
    if((i+1) % 10 == 0):
        print("Time: %6.0f [ns]     Rg: %6.1f [A]" % \
                  ((i+1)*n_inner_cycles*time_step_fs/1E6, Rg))
        if not DISABLE_RMF:
            os.update_always("Rg = %.1f A" % Rg)

# ==
# VII. report radius of gyration statistics
# ==
VarRg2= ERg4 - ERg2**2
StddevRg2= math.sqrt(VarRg2)
StderrRg2= StddevRg2/math.sqrt(n_outer_cycles)
mean_Rg= math.sqrt(ERg2)
mean_Rg_min= math.sqrt(ERg2 - 1.96*StderrRg2)
mean_Rg_max= math.sqrt(ERg2 + 1.96*StderrRg2)
try:
    Rg_min= math.sqrt(ERg2 - 1.96*StddevRg2)
except:
    Rg_min= 0.0 # undetermined
Rg_max= math.sqrt(ERg2 + 1.96*StddevRg2)
print("==================")
print("sqrt(<Rg2>) stats:")
print("==================")
print("Mean:  %.1f A" % mean_Rg)
print("Confidence interval: %.1f to %.1f A" % (mean_Rg_min, mean_Rg_max))
print("Rg typical fluctuations: %.1f to %.1f A" % (Rg_min, Rg_max))
print("Final score: %.1f kcal/mol" % scoring_function.evaluate(False))

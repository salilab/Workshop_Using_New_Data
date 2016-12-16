"""Use physical and NOE/dihedral restraints to model ubiquitin."""
import sys
import csv

import IMP
import IMP.isd
import IMP.pmi
import IMP.pmi.topology
import IMP.pmi.dof
import IMP.pmi.restraints
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.tools
import IMP.pmi.macros
from pmi_restraints import CharmmForceFieldRestraint, DihedralsRestraint, \
                           NOEsRestraint


if len(sys.argv) > 1:
    out_dir = sys.argv[1]
else:
    out_dir = "output"


def read_noes_file(fn):
    """Read distances predicted from NOEs as NOEs from file."""
    selection_tuples = []
    noes = []
    with open(fn, "rt") as f:
        for r in csv.reader(f, delimiter="\t"):
            sel_tups = [(int(r[i]), r[i + 1]) for i in range(0, 4, 2)]
            noe = float(r[-1]) ** (-6)
            selection_tuples.append(sel_tups)
            noes.append(noe)
    return selection_tuples, noes


def read_dihedrals_file(fn):
    """Read dihedrals predicted from J-coupling data from file."""
    selection_tuples = []
    angles = []
    with open(fn, "rt") as f:
        for r in csv.reader(f, delimiter="\t"):
            sel_tups = [(int(r[i]), r[i + 1]) for i in range(0, 8, 2)]
            angle = float(r[-1])
            selection_tuples.append(sel_tups)
            angles.append(angle)
    return selection_tuples, angles


# read input files
pdb_file = IMP.isd.get_example_path('ubiquitin/1G6J_MODEL1.pdb')
seqs = IMP.pmi.topology.Sequences("data/1G6J_seq.fa")
tuple_selection_pairs, noes = read_noes_file("data/noes.tab")
tuple_selection_quads, dihedrals = read_dihedrals_file("data/dihedrals.tab")

# set up system
m = IMP.Model()
s = IMP.pmi.topology.System(m)
st = s.create_state()
ubq = st.create_molecule("1G6J", seqs["1G6J"])
atomic = ubq.add_structure(pdb_file, chain_id='A')
ubq.add_representation(atomic, resolutions=[0])
hier = s.build()


all_rs = []
output_rs = []
# add NOE restraints
rnoes = NOEsRestraint(tuple_selection_pairs, noes, hier)
rnoes.add_to_model()
all_rs.append(rnoes)
output_rs.append(rnoes)

# add dihedral restraints
rdihed = DihedralsRestraint(tuple_selection_quads, dihedrals, hier)
rdihed.add_to_model()
all_rs.append(rdihed)
output_rs.append(rdihed)

# add CHARMM restraints
rcharmm = CharmmForceFieldRestraint(hier)
rcharmm.add_to_model()
all_rs.append(rcharmm)

# set up simulation
dof = IMP.pmi.dof.DegreesOfFreedom(m)
dof.get_nuisances_from_restraint(rnoes)
dof.get_nuisances_from_restraint(rdihed)

# minimize via md
md_ps = dof.setup_md(ubq)
IMP.set_log_level(IMP.SILENT)
# run replica exchange md with monte carlo
rex = IMP.pmi.macros.ReplicaExchange0(m,
                                      root_hier=hier,
                                      crosslink_restraints=output_rs,
                                      output_objects=all_rs,
                                      number_of_frames=100000,
                                      monte_carlo_sample_objects=dof.get_movers(),
                                      monte_carlo_steps=10,
                                      molecular_dynamics_sample_objects=md_ps,
                                      molecular_dynamics_steps=10,
                                      replica_exchange_minimum_temperature=1.,
                                      replica_exchange_maximum_temperature=2.5,
                                      number_of_best_scoring_models=0,
                                      atomistic=True,
                                      global_output_directory=out_dir)
rex.execute_macro()  # run the simulation

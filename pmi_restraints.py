"""PMI-style restraints."""
import math
import IMP
import IMP.isd
import IMP.core
import IMP.container
import IMP.atom
import IMP.pmi
import IMP.pmi.tools


class NOEsRestraint(object):

    """Restrain a protein using one or more NOEs."""

    def __init__(self, tuple_selection_pairs, noes, root_hier):
        self.m = root_hier.get_model()
        # create restraint sets to join restraints
        self.rs = IMP.RestraintSet(self.m, "likelihood")
        self.rs_priors = IMP.RestraintSet(self.m, "priors")

        # create nuisance particles
        self.gamma = IMP.pmi.tools.SetupNuisance(
            self.m, 1., 0.01, 100., isoptimized=True).get_particle()
        self.sigma = IMP.pmi.tools.SetupNuisance(
            self.m, 1., 0.01, 100., isoptimized=True).get_particle()

        # create NOE likelihood restraints
        for (tuple1, tuple2), noe in zip(tuple_selection_pairs, noes):
            # get atom 1
            sel1 = IMP.atom.Selection(root_hier,
                                      resolution=0,
                                      residue_index=tuple1[0],
                                      atom_type=IMP.atom.AtomType(tuple1[1]))
            p1 = sel1.get_selected_particles()[0]
            # get atom 2
            sel2 = IMP.atom.Selection(root_hier,
                                      resolution=0,
                                      residue_index=tuple2[0],
                                      atom_type=IMP.atom.AtomType(tuple2[1]))
            p2 = sel2.get_selected_particles()[0]

            # create restraint and add to set
            r = IMP.isd.NOERestraint(self.m, p1, p2, self.sigma, self.gamma,
                                     noe)
            r.set_name("NOERestraint_{0}:{1}_{2}:{3}".format(tuple1[0],
                                                             tuple1[1],
                                                             tuple2[0],
                                                             tuple2[1]))
            self.rs.add_restraint(r)

        # create prior restraints
        self.rs_priors.add_restraint(IMP.isd.JeffreysRestraint(self.m,
                                                               self.gamma))
        self.rs_priors.add_restraint(IMP.isd.JeffreysRestraint(self.m,
                                                               self.sigma))

    def add_to_model(self):
        """Add the restraints to the model."""
        for rs in [self.rs, self.rs_priors]:
            IMP.pmi.tools.add_restraint_to_model(self.m, rs)

    def get_restraint(self):
        return self.rs

    def get_restraint_for_rmf(self):
        """Get the restraint for visualization in an RMF file."""
        return self.rs

    def get_particles_to_sample(self):
        """Get any created particles which should be sampled."""
        out = {}
        out["Nuisances_Gamma"] = ([self.gamma], .1)
        out["Nuisances_Sigma"] = ([self.sigma], .1)
        return out

    def get_output(self):
        """Get outputs to write to stat files."""
        output = {}
        self.m.update()
        likelihood_score = self.rs.unprotected_evaluate(None)
        prior_score = self.rs_priors.unprotected_evaluate(None)
        output["_TotalScore"] = likelihood_score + prior_score
        output["NOELikelihood_Score"] = likelihood_score
        output["NOEPrior_Score"] = prior_score
        output["NOE_Gamma"] = self.gamma.get_scale()
        output["NOE_Sigma"] = self.sigma.get_scale()
        return output


class DihedralsRestraint(object):

    """Restrain a protein using one or more predicted dihedrals.

    Dihedrals are predicted from J-coupling data. All dihedrals are
    assumed to have equal error/uncertainty.
    """

    def __init__(self, tuple_selection_quads, dihedrals, root_hier):
        self.m = root_hier.get_model()
        # create restraint sets to join restraints
        self.rs = IMP.RestraintSet(self.m, "likelihood")
        self.rs_priors = IMP.RestraintSet(self.m, "priors")

        # create nuisance particles
        self.kappa = IMP.pmi.tools.SetupNuisance(
            self.m, 1., 1e-5, 1e5, isoptimized=True).get_particle()

        # create NOE likelihood restraints
        for (tuple1, tuple2, tuple3, tuple4), angle in zip(
                tuple_selection_quads, dihedrals):
            # get atom 1
            p1 = self.get_atom_from_selection(root_hier, tuple1)
            p2 = self.get_atom_from_selection(root_hier, tuple2)
            p3 = self.get_atom_from_selection(root_hier, tuple3)
            p4 = self.get_atom_from_selection(root_hier, tuple4)

            # create restraint and add to set
            angle = angle * math.pi / 180.  # convert to radians
            r = IMP.isd.TALOSRestraint(self.m, p1, p2, p3, p4, [angle],
                                       self.kappa)
            r.set_name(
                "DihedralRestraint_{0}:{1}_{2}:{3}_{4}:{5}_{6}:{7}".format(
                    tuple1[0], tuple1[1], tuple2[0], tuple2[1],
                    tuple3[0], tuple3[1], tuple4[0], tuple4[1]))
            self.rs.add_restraint(r)

        # create prior restraints
        self.rs_priors.add_restraint(IMP.isd.vonMisesKappaJeffreysRestraint(
            self.m, self.kappa))

    def get_atom_from_selection(self, root_hier, sel_tuple):
        res_id, atom_name = sel_tuple
        sel = IMP.atom.Selection(root_hier,
                                 resolution=0,
                                 residue_index=res_id,
                                 atom_type=IMP.atom.AtomType(atom_name))
        return sel.get_selected_particles()[0]

    def add_to_model(self):
        """Add the restraints to the model."""
        for rs in [self.rs, self.rs_priors]:
            IMP.pmi.tools.add_restraint_to_model(self.m, rs)

    def get_restraint(self):
        return self.rs

    def get_restraint_for_rmf(self):
        """Get the restraint for visualization in an RMF file."""
        return self.rs

    def get_particles_to_sample(self):
        """Get any created particles which should be sampled."""
        out = {}
        out["Nuisances_Kappa"] = ([self.kappa], .1)
        return out

    def get_output(self):
        """Get outputs to write to stat files."""
        output = {}
        self.m.update()
        likelihood_score = self.rs.unprotected_evaluate(None)
        prior_score = self.rs_priors.unprotected_evaluate(None)
        output["_TotalScore"] = likelihood_score + prior_score
        output["DihedralLikelihood_Score"] = likelihood_score
        output["DihedralPrior_Score"] = prior_score
        output["Dihedral_Kappa"] = self.kappa.get_scale()
        return output


class CharmmForceFieldRestraint(object):
    """ Enable CHARMM force field.

    Modified from
    IMP.pmi.restraints.stereochemistry.CharmmForceFieldRestraint
    """
    def __init__(self,
                 root,
                 ff_temp=300.0):
        """Setup the CHARMM restraint on a selection. Expecting atoms.
        @param root             The node at which to apply the restraint
        @param ff_temp          The temperature of the force field
        """
        kB = (1.381 * 6.02214) / 4184.0
        self.m = root.get_model()
        self.bonds_rs = IMP.RestraintSet(self.m, 1.0 / (kB * ff_temp), 'BONDED')
        self.nonbonded_rs = IMP.RestraintSet(self.m, 1.0 / (kB * ff_temp), 'NONBONDED')
        self.weight = 1.0
        self.label = ""

        ### setup topology and bonds etc
        ff = IMP.atom.get_all_atom_CHARMM_parameters()
        topology = ff.create_topology(root)
        topology.apply_default_patches()
        topology.setup_hierarchy(root)
        r = IMP.atom.CHARMMStereochemistryRestraint(root, topology)
        self.ps = IMP.core.get_leaves(root)
        print('init bonds score',r.unprotected_evaluate(None))
        self.bonds_rs.add_restraint(r)
        ff.add_radii(root)
        ff.add_well_depths(root)

        atoms = IMP.atom.get_by_type(root,IMP.atom.ATOM_TYPE)
        ### non-bonded forces
        cont = IMP.container.ListSingletonContainer(self.m,atoms)
        self.nbl = IMP.container.ClosePairContainer(cont, 4.0)
        self.nbl.add_pair_filter(r.get_full_pair_filter())
        pairscore = IMP.isd.RepulsiveDistancePairScore(0,1)
        pr=IMP.container.PairsRestraint(pairscore, self.nbl)
        self.nonbonded_rs.add_restraint(pr)
        print('CHARMM is set up')

    def set_label(self, label):
        self.label = label
        self.rs.set_name(label)
        for r in self.rs.get_restraints():
            r.set_name(label)

    def add_to_model(self):
        IMP.pmi.tools.add_restraint_to_model(self.m, self.bonds_rs)
        IMP.pmi.tools.add_restraint_to_model(self.m, self.nonbonded_rs)

    def get_restraint(self):
        return self.rs

    def get_close_pair_container(self):
        return self.nbl

    def set_weight(self, weight):
        self.weight = weight
        self.rs.set_weight(weight)

    def get_output(self):
        self.m.update()
        output = {}
        bonds_score = self.weight * self.bonds_rs.unprotected_evaluate(None)
        nonbonded_score = self.weight * self.nonbonded_rs.unprotected_evaluate(None)
        score=bonds_score+nonbonded_score
        output["_TotalScore"] = str(score)
        output["CHARMM_BONDS"] = str(bonds_score)
        output["CHARMM_NONBONDED"] = str(nonbonded_score)
        return output

"""IMP.Restraint style restraints that we'll need."""
import math
import IMP
import IMP.isd
import IMP.core


class JeffreysRestraint(IMP.Restraint):
    """Jeffreys prior on the sigma parameter of a normal distribution."""
    def __init__(self, m, s):
        IMP.Restraint.__init__(self, m, "JeffreysRestraint%1%")
        self.s = s

    def do_add_score_and_derivatives(self, sa):
        sig = IMP.isd.Scale(self.get_model(), self.s)
        score = math.log(sig.get_scale())
        if sa.get_derivative_accumulator():
            deriv = 1. / sig.get_scale()
            sig.add_to_scale_derivative(deriv, sa.get_derivative_accumulator())
        sa.add_score(score)

    def do_get_inputs(self):
        return [self.get_model().get_particle(self.s)]


class NOERestraint(IMP.Restraint):
    """Apply an NOE distance restraint between two particles."""
    def __init__(self, m, p0, p1, sigma, gamma, Iexp):
        IMP.Restraint.__init__(self, m, "NOERestraint%1%")
        pass

    def do_add_score_and_derivatives(self, sa):
        pass

    def do_get_inputs(self):
        pass

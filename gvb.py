import numpy, scipy

import pyscf
from pyscf import gto, scf, ao2mo, lib

from pyscf.mcscf.mc1step import CASSCF
from pp import PerfectPairingAsFCISolver

einsum = numpy.einsum

def _fake_h(gvb_obj, mo_coeff, eris):
    ncas = gvb_obj.ncas
    ncore = gvb_obj.ncore
    nocc = ncore + ncas
    nmo = mo_coeff.shape[0]
    assert mo_coeff.shape == (nmo, ncas)
    
    

class GeneralizedValenceBondTheory(CASSCF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fcisolver = PerfectPairingAsFCISolver(
            self.mol, singlet=None, symm=False
        )

        ncas = self.ncas
        assert ncas % 2 == 0
        npair = ncas // 2

        # neccessary for perfect pairing
        assert self.nelecas == (npair, npair)

    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        raise NotImplementedError

GVB = GeneralizedValenceBondTheory

if __name__ == '__main__':
    import sys
    from pyscf import gto
    from automr import guess
    from automr.autocas import get_uno, loc_asrot

    sys.path.append("..")
    lib.num_threads(4)

    mol = gto.Mole(
        atom='''
            C -2.94294278    0.39039038    0.00000000
            C -1.54778278    0.39039038    0.00000000
            C -0.85024478    1.59814138    0.00000000
            C -1.54789878    2.80665038   -0.00119900
            C -2.94272378    2.80657238   -0.00167800
            C -3.64032478    1.59836638   -0.00068200
            H -3.49270178   -0.56192662    0.00045000
            H -0.99827478   -0.56212262    0.00131500
            H  0.24943522    1.59822138    0.00063400
            H -0.99769878    3.75879338   -0.00125800
            H -3.49284578    3.75885338   -0.00263100
            H -4.73992878    1.59854938   -0.00086200
        ''',
        basis='sto3g',
        verbose=0
    )

    mf = guess._mix(mol, conv='tight', newton=True)
    mf.verbose = 0

    mf, unos, noon, nacto, nelecact, ncore, _ = get_uno(mf, thresh=1.98)
    mf, _ = loc_asrot(mf, nacto, nelecact, ncore)

    gvb_obj = GVB(mf, nacto, nelecact)
    gvb_obj.verbose = 10
    gvb_obj.kernel()

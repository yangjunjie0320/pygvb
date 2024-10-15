"""Module for electronic energy calculations in GVB methods."""

import numpy
from pyscf import lib
from pyscf.lib import logger

einsum = numpy.einsum
def energy_func(t, h1e=None, j1e=None, k1e=None):
    npair = len(t)
    norb = 2 * npair
    assert h1e.shape == (norb, norb)
    assert j1e.shape == (norb, norb)
    assert k1e.shape == (norb, norb)

    jk = 2.0 * j1e - k1e
    hdd = numpy.diag(h1e)
    kdd = numpy.diag(k1e)
    kdf = numpy.diag(numpy.flip(k1e, axis=0))
    jkdd = numpy.diag(jk)
    jkdf = numpy.diag(numpy.flip(jk, axis=0))

    x = (1.0 + numpy.sin(t) ** 2) / 2.0
    x2 = x ** 2

    f = numpy.hstack((x2, (1.0 - x2)[::-1])) * 2.0
    assert numpy.all(0.0 <= f) and numpy.all(f <= 2.0)

    e = numpy.sum(f * hdd)
    e += einsum("p,q,qp->", f, f, jk, optimize=True) * 0.25

    f1 = f * 0.5
    e += numpy.sum(f1 * kdd)
    e -= numpy.sum(f1 * f1 * jkdd)

    f2 = numpy.sqrt(f * f[::-1]) * 0.5
    e -= numpy.sum(f2 * kdf)
    e -= numpy.sum(f2 * f2 * jkdf)

    return e

def energy_grad(t, h1e=None, j1e=None, k1e=None):
    npair = len(t)
    norb = 2 * npair
    assert h1e.shape == (norb, norb)
    assert j1e.shape == (norb, norb)
    assert k1e.shape == (norb, norb)

    jk = 2.0 * j1e - k1e
    hdd = numpy.diag(h1e)
    kdd = numpy.diag(k1e)
    kdf = numpy.diag(numpy.flip(k1e, axis=0))
    jkdd = numpy.diag(jk)
    jkdf = numpy.diag(numpy.flip(jk, axis=0))

    x = (1.0 + numpy.sin(t) ** 2) / 2.0
    x2 = x ** 2
    f = numpy.hstack((x2, (1.0 - x2)[::-1])) * 2.0
    assert numpy.all(0.0 <= f) and numpy.all(f <= 2.0)

    dx_dt = numpy.sin(t) * numpy.cos(t)
    dx2_dt = 2 * x * dx_dt
    dx2_dt = numpy.diag(dx2_dt)

    df_dt = 2 * numpy.vstack((dx2_dt, -dx2_dt[::-1]))
    assert df_dt.shape == (norb, npair)

    e = einsum("p,p->", f, hdd) # * 2.0
    e += einsum("p,q,qp->", f, f, jk, optimize=True) * 0.25

    f1 = f * 0.5
    e += numpy.sum(f1 * kdd)
    e -= numpy.sum(f1 * f1 * jkdd)

    f2 = numpy.sqrt(f * f[::-1]) * 0.5
    e -= numpy.sum(f2 * kdf)
    e -= numpy.sum(f2 * f2 * jkdf)

    de_dt = einsum("pk,p->k", df_dt, hdd) # * 2.0
    de_dt += einsum("pk,q,qp->k", df_dt, f, jk, optimize=True) * 0.25
    de_dt += einsum("p,qk,qp->k", f, df_dt, jk, optimize=True) * 0.25

    df1_dt = df_dt * 0.5
    assert df1_dt.shape == (norb, npair)
    de_dt += numpy.einsum("pk,p->k", df1_dt, kdd)
    de_dt -= numpy.einsum("pk,p,p->k", df1_dt, f1, jkdd)
    de_dt -= numpy.einsum("p,pk,p->k", f1, df1_dt, jkdd)

    f2 = numpy.sqrt(f * f[::-1]) * 0.5 + 1e-10
    df2_dt = df1_dt[::-1] * f[:, None] + df1_dt * f[::-1, None]
    df2_dt /= (4 * f2[:, None])

    de_dt -= numpy.einsum("pk,p->k", df2_dt, kdf)
    de_dt -= numpy.einsum("p,pk,p->k", f2, df2_dt, jkdf)
    de_dt -= numpy.einsum("pk,p,p->k", df2_dt, f2, jkdf)
    return de_dt

class PerfectPairingAsFCISolver(lib.StreamObject):
    scale_f0 = 0.9
    maxstep = 0.05
    method = 'BFGS'

    max_cycle = 200
    conv_tol = 1e-8
    grad_tol = 1e-5

    def __init__(self, mol=None, singlet=None, symm=False):
        assert singlet is None
        assert symm is False
        self.mol = mol

    def gen_minimize(self, verbose=0):
        options = {
            'gtol': self.grad_tol,
            'maxiter': self.max_cycle,
            'disp': False
        }

        method = self.method
        assert method in ['BFGS', 'L-BFGS-B']

        log = logger.new_logger(self, verbose)
        log.info("\nSetting up the minimization method: %s", method)

        import scipy.optimize
        def minimize(x0, func=None, grad=None):
            res = scipy.optimize.minimize(
                func, x0, jac=grad, method=method,
                tol=self.conv_tol, options=options
            )

            log.info("%s", res.message)
            log.info("x: %s" % res.x + ", fun: % 12.8f" % res.fun)
            log.info("nit = %d, nfev = %d, njev = %d\n", res.nit, res.nfev, res.njev)
            return res
        
        return minimize

    def kernel(self, h1e, h2e, norb, nelec, ci0=None, t0=None, verbose=0,
               max_memory=None, ecore=None):
        npair = norb // 2
        assert npair * 2 == norb

        if ci0 is not None:
            assert t0 is None
            f0 = ci0

        if t0 is None:
            f0 = numpy.ones(npair) * self.scale_f0
            t0 = (2.0 * f0 ** 0.5 - 1) ** 0.5
            t0 = numpy.arcsin(t0)
        
        assert t0.shape == (npair, )

        j1e, k1e = h2e
        assert j1e.shape == (norb, norb)
        assert k1e.shape == (norb, norb)

        log = logger.new_logger(self, verbose)
        log.info("\nt0 = %s", t0)

        global icycle
        icycle = 0

        def func(t):
            global icycle
            icycle += 1
            e = energy_func(t, h1e, j1e, k1e)
            log.info("cycle %3d: e = % 12.8f", icycle, e)
            return e
        
        def grad(t):
            return energy_grad(t, h1e, j1e, k1e)

        minimize = self.gen_minimize(verbose=log)
        res = minimize(t0, func, grad)

        t = res.x
        e = res.fun
        return e + ecore, t

    def make_rdm12(self, t=None, ncas=None, nelec=None):
        assert ncas % 2 == 0

        x = (1.0 + numpy.sin(t) ** 2) / 2.0
        x2 = x ** 2
        f = numpy.hstack((x2, (1.0 - x2)[::-1])) * 2.0
        assert numpy.all(0.0 <= f) and numpy.all(f <= 2.0)

        rdm1 = numpy.diag(f)
        assert rdm1.shape == (ncas, ncas)

        f1 = numpy.diag(f) * 0.5
        f2 = numpy.sqrt(f * f[::-1]) * 0.5
        f2 = numpy.flip(numpy.diag(f2), axis=0)

        p = f1 - f2
        d = f1 ** 2 + f2 ** 2

        a = einsum('q,p->qp', f, f) * 0.5 - d * 2.0
        b = p - a * 0.5

        rdm2 = numpy.zeros((ncas, ncas, ncas, ncas))
        for p in range(ncas):
            for q in range(ncas):
                rdm2[p, p, q, q] += a[p, q]
                rdm2[p, q, q, p] += b[p, q]
        return rdm1, rdm2 * 2.0
    
    def make_rdm1(self, t, ncas, nelec):
        assert ncas % 2 == 0

        x = (1.0 + numpy.sin(t) ** 2) / 2.0
        x2 = x ** 2
        f = numpy.hstack((x2, (1.0 - x2)[::-1])) * 2.0
        assert numpy.all(0.0 <= f) and numpy.all(f <= 2.0)

        rdm1 = numpy.diag(f)
        assert rdm1.shape == (ncas, ncas)
        return rdm1

from pyscf.mcscf.casci import CASCI
class PerfectPairing(CASCI):
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

    def get_h2eff(self, mo_coeff):
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas

        if mo_coeff is None:
            ncore = self.ncore
            mo_coeff = self.mo_coeff[:,ncore:nocc]

        elif mo_coeff.shape[1] != ncas:
            mo_coeff = mo_coeff[:,ncore:nocc]

        nao = mo_coeff.shape[0]
        assert mo_coeff.shape == (nao, ncas)

        dms = einsum("mp,np->pmn", mo_coeff, mo_coeff)
        assert dms.shape == (ncas, nao, nao)

        j, k = self._scf.get_jk(dm=dms)
        j1e = einsum("qmn,mp,np->pq", j, mo_coeff, mo_coeff, optimize=True)
        k1e = einsum("pnk,nq,kq->pq", k, mo_coeff, mo_coeff, optimize=True)
        
        assert j1e.shape == (ncas, ncas)
        assert k1e.shape == (ncas, ncas)
        return (j1e, k1e)
    
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

    from pyNOF import nof
    nof_obj = nof.SOPNOF(mf, nacto, nelecact)
    nof_obj.verbose = 0
    nof_obj.fcisolver = nof.fakeFCISolver()
    nof_obj.fcisolver.verbose = 0
    nof_obj.fcisolver.ncore = ncore
    nof_obj.fcisolver.npair = nacto // 2
    nof_obj.internal_rotation = True  # important for this case!
    nof_obj.max_cycle_macro = 30
    nof_obj.max_stepsize = 0.05  # increase stepsize helps in this case
    nof_obj.mc2step()

    f0 = nof_obj.mo_occ
    ncas = nof_obj.ncas
    assert ncas % 2 == 0
    npair = ncas // 2
    assert nof_obj.nelecas == (npair, npair)

    rdm1_ref, rdm2_ref = nof_obj.fcisolver.make_rdm12(f0, ncas, nof_obj.nelecas)

    mf.mo_coeff = nof_obj.mo_coeff
    pp_obj = PerfectPairing(mf, ncas, nof_obj.nelecas)
    pp_obj.verbose = 10
    pp_obj.canonicalization = True
    ene_sol = pp_obj.kernel()[0]
    assert abs(ene_sol + 227.94090761926827) < 1e-5

    # from pyscf.mcscf import mc1step
    # coeff = nof_obj.mo_coeff
    # nmo = coeff.shape[1]
    # eris = nof_obj.ao2mo(coeff)

    # heff = mc1step._fake_h_for_fast_casci(nof_obj, coeff, eris=eris)
    # h1e, e0 = heff.get_h1eff(coeff)
    # h2e = heff.get_h2eff(coeff)

    # norb = h1e.shape[0]
    # npair = norb // 2
    # nelec = norb
    # nopen = 0
    # assert h1e.shape == (norb, norb)
    # assert h2e.shape == (norb, norb, norb, norb)

    # j1e = numpy.einsum("iijj->ij", h2e)
    # k1e = numpy.einsum("ijji->ij", h2e)

    # # check the gradient
    # for i in range(10):
    #     from scipy.optimize import check_grad
    #     t1 = numpy.random.rand(npair)
    #     err = check_grad(
    #         lambda t: energy_func(t, h1e, j1e, k1e),
    #         lambda t: energy_grad(t, h1e, j1e, k1e),
    #         t1
    #     )
    #     assert err < 1e-5, "Gradient does not match"

    # pp_obj = nof_obj.fcisolver.nof
    # e1, e2 = pp_obj.kernel(
    #     h1e, h2e, mo=coeff, mo_occ=None,
    #     iter_occ=True
    # )
    
    # f_ref = pp_obj.mo_occ # * 2.0

    # from nof import t2X, get_DP, energy_elec
    # t0 = numpy.array([1.502738, 1.38949, 1.389455])
    # x0 = t2X(t0)

    # occ = numpy.zeros(nmo)
    # occ[:ncore] = 1.0
    # occ[ncore:ncore+npair] = x0 ** 2
    # occ[ncore+npair:ncore+norb] = (1.0 - x0 ** 2)[::-1]
    # assert numpy.allclose(occ, f_ref)
    
    # occ = occ
    # d, p = get_DP(occ, ncore, npair, nopen)
    # nact = npair * 2 + nopen
    # e_ref = energy_elec(occ, ncore, nact, h1e, j1e, k1e, d, p)
    # print(e_ref)

    # pp_obj = PerfectPairing()
    # print(t0)

    # t_sol, e_sol, f_sol = pp_obj.kernel(
    #     h1e, h2e, norb, nelec,
    #     f0=None
    # )

    # e_sol = energy_func(t0, h1e, j1e, k1e)
    # assert numpy.allclose(e_sol, e_ref), "Energy does not match"

    # # print(e_sol)
    # # print(t_sol)
    # # print(f_sol)

    # # assert 1 == 2

    # e_sol = 0.0
    # for i in range(npair):
    #     tt = numpy.zeros(npair)
    #     tt[i] = t0[i]

    #     ff = numpy.zeros(norb)
    #     p = i
    #     q = norb - i - 1
    #     print(p, q)
    #     ff[p] = f_ref[ncore + p]
    #     ff[q] = f_ref[ncore + q]

    #     ei = energy_func(tt, ff * 2.0, h1e, j1e, k1e)
    #     print(ei)
    #     e_sol += ei

    # print(e_sol)
    # e_sol = energy_func(t0, h1e, j1e, k1e)
    # assert abs(e_sol - e_ref) < 1e-5

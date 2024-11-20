# -*- coding: utf-8 -*-
"""
https://github.com/brunobeltran/mittag-leffler
Roberto Garrappa (2020). The Mittag-Leffler function (https://www.mathworks.com/matlabcentral/fileexchange/48154-the-mittag-leffler-function), MATLAB Central File Exchange. Retrieved April 7, 2020.
[1] R. Garrappa, Numerical evaluation of two and three parameter Mittag-Leffler functions, SIAM Journal of Numerical Analysis, 2015, 53(3), 1350-1369
E = ML(z,alpha,beta,gama) evaluates the ML function with three parameters alpha,
beta and gama for the corresponding elements of z; alpha must be a real scalar
such that 0<alpha<1, beta any real scalar and gama a real and positive scalar;
the arguments z must satisfy |Arg(z)| > alpha*pi.
"""
import numpy as np
from scipy.special import gamma


def LTInversion(t,lamda,alpha,beta,gama,log_epsilon):
    # Evaluation of the relevant poles
    theta = np.angle(lamda)
    kmin = np.ceil(-alpha/2. - theta/2./np.pi)
    kmax = np.floor(alpha/2. - theta/2./np.pi)
    k_vett = np.arange(kmin, kmax+1)
    s_star = np.abs(lamda)**(1./alpha) * np.exp(1j*(theta+2*k_vett*np.pi)/alpha)

    # Evaluation of phi(s_star) for each pole
    phi_s_star = (np.real(s_star)+np.abs(s_star))/2

    # Sorting of the poles according to the value of phi(s_star)
    index_s_star = np.argsort(phi_s_star)
    phi_s_star = phi_s_star.take(index_s_star)
    s_star = s_star.take(index_s_star)

    # Deleting possible poles with phi_s_star=0
    index_save = phi_s_star > 1.0e-15
    s_star = s_star.repeat(index_save)
    phi_s_star = phi_s_star.repeat(index_save)

    # Inserting the origin in the set of the singularities
    s_star = np.hstack([[0], s_star])
    phi_s_star = np.hstack([[0], phi_s_star])
    J1 = len(s_star)
    J = J1 - 1

    # Strength of the singularities
    p = gama*np.ones((J1,), np.float64)
    p[0] = max(0,-2*(alpha*gama-beta+1))
    q = gama*np.ones((J1,), np.float64)
    q[-1] = np.inf
    phi_s_star = np.hstack([phi_s_star, [np.inf]])

    # Looking for the admissible regions with respect to round-off errors
    admissible_regions = \
       np.nonzero(np.bitwise_and(
           (phi_s_star[:-1] < (log_epsilon - np.log(np.finfo(np.float64).eps))/t),
           (phi_s_star[:-1] < phi_s_star[1:])))[0]
    # Initializing vectors for optimal parameters
    JJ1 = admissible_regions[-1]
    mu_vett = np.ones((JJ1+1,), np.float64)*np.inf
    N_vett = np.ones((JJ1+1,), np.float64)*np.inf
    h_vett = np.ones((JJ1+1,), np.float64)*np.inf

    # Evaluation of parameters for inversion of LT in each admissible region
    find_region = False
    while not find_region:
        for j1 in admissible_regions:
            if j1 < J1-1:
                muj, hj, Nj = OptimalParam_RB(t, phi_s_star[j1], phi_s_star[j1+1], p[j1], q[j1], log_epsilon)
            else:
                muj, hj, Nj = OptimalParam_RU(t, phi_s_star[j1], p[j1], log_epsilon)
            mu_vett[j1] = muj
            h_vett[j1] = hj
            N_vett[j1] = Nj
        if N_vett.min() > 200:
            log_epsilon = log_epsilon + np.log(10)
        else:
            find_region = True

    # Selection of the admissible region for integration which
    # involves the minimum number of nodes
    iN = np.argmin(N_vett)
    N = N_vett[iN]
    mu = mu_vett[iN]
    h = h_vett[iN]

    # Evaluation of the inverse Laplace transform
    k = np.arange(-N, N+1)
    u = h*k
    z = mu*(1j*u+1.)**2
    zd = -2.*mu*u + 2j*mu
    zexp = np.exp(z*t)
    F = z**(alpha*gama-beta)/(z**alpha - lamda)**gama*zd
    S = zexp*F
    Integral = h*np.sum(S)/2./np.pi/1j

    # Evaluation of residues
    ss_star = s_star[iN+1:]
    Residues = np.sum(1./alpha*(ss_star)**(1-beta)*np.exp(t*ss_star))

    # Evaluation of the ML function
    E = Integral + Residues
    if np.imag(lamda) == 0.:
        E = np.real(E)
    return E

# =========================================================================
# Finding optimal parameters in a right-bounded region
# =========================================================================
def OptimalParam_RB(t, phi_s_star_j, phi_s_star_j1, pj, qj, log_epsilon):
    # Definition of some constants
    log_eps = -36.043653389117154 # log(eps)
    fac = 1.01
    conservative_error_analysis = False

    # Maximum value of fbar as the ration between tolerance and round-off unit
    f_max = np.exp(log_epsilon - log_eps)

    # Evaluation of the starting values for sq_phi_star_j and sq_phi_star_j1
    sq_phi_star_j = np.sqrt(phi_s_star_j)
    threshold = 2.*np.sqrt((log_epsilon - log_eps)/t)
    sq_phi_star_j1 = min(np.sqrt(phi_s_star_j1), threshold - sq_phi_star_j)

    # Zero or negative values of pj and qj
    if pj < 1.0e-14 and qj < 1.0e-14:
        sq_phibar_star_j = sq_phi_star_j
        sq_phibar_star_j1 = sq_phi_star_j1
        adm_region = 1

    # Zero or negative values of just pj
    if pj < 1.0e-14 and qj >= 1.0e-14:
        sq_phibar_star_j = sq_phi_star_j
        if sq_phi_star_j > 0:
            f_min = fac*(sq_phi_star_j/(sq_phi_star_j1-sq_phi_star_j))**qj
        else:
            f_min = fac
        if f_min < f_max:
            f_bar = f_min + f_min/f_max*(f_max-f_min)
            fq = f_bar**(-1/qj)
            sq_phibar_star_j1 = (2*sq_phi_star_j1-fq*sq_phi_star_j)/(2+fq)
            adm_region = True
        else:
            adm_region = False

    # Zero or negative values of just qj
    if pj >= 1.0e-14 and qj < 1.0e-14:
        sq_phibar_star_j1 = sq_phi_star_j1
        f_min = fac*(sq_phi_star_j1/(sq_phi_star_j1-sq_phi_star_j))**pj
        if f_min < f_max:
            f_bar = f_min + f_min/f_max*(f_max-f_min)
            fp = f_bar**(-1./pj)
            sq_phibar_star_j = (2.*sq_phi_star_j+fp*sq_phi_star_j1)/(2-fp)
            adm_region = True
        else:
            adm_region = False

    # Positive values of both pj and qj
    if pj >= 1.0e-14 and qj >= 1.0e-14:
        f_min = fac*(sq_phi_star_j+sq_phi_star_j1) / \
                (sq_phi_star_j1-sq_phi_star_j)**max(pj, qj)
        if f_min < f_max:
            f_min = max(f_min,1.5)
            f_bar = f_min + f_min/f_max*(f_max-f_min)
            fp = f_bar**(-1/pj)
            fq = f_bar**(-1/qj)
            if ~conservative_error_analysis:
                w = -phi_s_star_j1*t/log_epsilon
            else:
                w = -2.*phi_s_star_j1*t/(log_epsilon-phi_s_star_j1*t)
            den = 2+w - (1+w)*fp + fq
            sq_phibar_star_j = ((2+w+fq)*sq_phi_star_j + fp*sq_phi_star_j1)/den
            sq_phibar_star_j1 = (-(1.+w)*fq*sq_phi_star_j + (2.+w-(1.+w)*fp)*sq_phi_star_j1)/den
            adm_region = True
        else:
            adm_region = False

    if adm_region:
        log_epsilon = log_epsilon  - np.log(f_bar)
        if not conservative_error_analysis:
            w = -sq_phibar_star_j1**2*t/log_epsilon
        else:
            w = -2.*sq_phibar_star_j1**2*t/(log_epsilon-sq_phibar_star_j1**2*t)
        muj = (((1.+w)*sq_phibar_star_j + sq_phibar_star_j1)/(2.+w))**2
        hj = -2.*np.pi/log_epsilon*(sq_phibar_star_j1-sq_phibar_star_j) \
             / ((1.+w)*sq_phibar_star_j + sq_phibar_star_j1)
        Nj = np.ceil(np.sqrt(1-log_epsilon/t/muj)/hj)
    else:
        muj = 0.
        hj = 0.
        Nj = np.inf

    return muj, hj, Nj


# =========================================================================
# Finding optimal parameters in a right-unbounded region
# =========================================================================
def OptimalParam_RU(t, phi_s_star_j, pj, log_epsilon):
    # Evaluation of the starting values for sq_phi_star_j
    sq_phi_s_star_j = np.sqrt(phi_s_star_j)
    if phi_s_star_j > 0:
        phibar_star_j = phi_s_star_j*1.01
    else:
        phibar_star_j = 0.01
    sq_phibar_star_j = np.sqrt(phibar_star_j)

    # Definition of some constants
    f_min = 1
    f_max = 10
    f_tar = 5

    # Iterative process to look for fbar in [f_min,f_max]
    while True:
        phi_t = phibar_star_j*t
        log_eps_phi_t = log_epsilon/phi_t
        Nj = np.ceil(phi_t/np.pi*(1. - 3*log_eps_phi_t/2 + np.sqrt(1-2*log_eps_phi_t)))
        A = np.pi*Nj/phi_t
        sq_muj = sq_phibar_star_j*np.abs(4-A)/np.abs(7-np.sqrt(1+12*A))
        fbar = ((sq_phibar_star_j-sq_phi_s_star_j)/sq_muj)**(-pj)
        if (pj < 1.0e-14) or (f_min < fbar and fbar < f_max):
            break
        sq_phibar_star_j = f_tar**(-1./pj)*sq_muj + sq_phi_s_star_j
        phibar_star_j = sq_phibar_star_j**2
    muj = sq_muj**2
    hj = (-3*A - 2 + 2*np.sqrt(1+12*A))/(4-A)/Nj

    # Adjusting integration parameters to keep round-off errors under control
    log_eps = np.log(np.finfo(np.float64).eps)
    threshold = (log_epsilon - log_eps)/t
    if muj > threshold:
        if abs(pj) < 1.0e-14:
            Q = 0
        else:
            Q = f_tar**(-1/pj)*np.sqrt(muj)
        phibar_star_j = (Q + np.sqrt(phi_s_star_j))**2
        if phibar_star_j < threshold:
            w = np.sqrt(log_eps/(log_eps-log_epsilon))
            u = np.sqrt(-phibar_star_j*t/log_eps)
            muj = threshold
            Nj = np.ceil(w*log_epsilon/2/np.pi/(u*w-1))
            hj = np.sqrt(log_eps/(log_eps - log_epsilon))/Nj
        else:
            Nj = np.inf
            hj = 0

    return muj, hj, Nj

def ml(z, alpha, beta=1., gama=1.):
    eps = np.finfo(np.float64).eps
    if np.real(alpha) <= 0 or np.real(gama) <= 0 or np.imag(alpha) != 0. \
       or np.imag(beta) != 0. or np.imag(gama) != 0.:
        raise ValueError('ALPHA and GAMA must be real and positive. BETA must be real.')
    if np.abs(gama-1.) > eps:
        if alpha > 1.:
            raise ValueError('GAMMA != 1 requires 0 < ALPHA < 1')
        if (np.abs(np.angle(np.repeat(z, np.abs(z) > eps))) <= alpha*np.pi).any():
            raise ValueError('|Arg(z)| <= alpha*pi')

    return np.vectorize(ml_, [np.float64])(z, alpha, beta, gama)

def ml_(z, alpha, beta, gama):
    # Target precision
    log_epsilon = np.log(1.e-15)
    # Inversion of the LT
    if np.abs(z) < 1.e-15:
        return 1/gamma(beta)
    else:
        return LTInversion(1, z, alpha, beta, gama, log_epsilon)

def mlf(alpha, beta, zarray):
    val =np.empty(zarray.shape)
    for ij in range(zarray.shape[0]):
        val[ij] = ml_(zarray[ij], alpha, beta, gama=1.)
    return val


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    x = np.arange(0, 10, 0.1)

    plt.figure(figsize=(10,5))
    for ii in range(5):
        plt.plot(x, mlf(ii*0.2+0.01, 1, -x*x), label="alpha = "+str(ii))
    plt.legend()
    plt.ylim(-0.1, 1.1); plt.xlim(-1, 11); plt.grid()
    # plt.plot(x, mlf(-x*x, 0.1, 1, 1))
    plt.figure(2)
    t = np.logspace(-5, 5, 100)
    test = mlf(0.7, 1, -(100/300)*t**0.7)
    plt.loglog(t, test)

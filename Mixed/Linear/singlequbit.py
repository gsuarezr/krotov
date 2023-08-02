import qutip
import numpy as np
import krotov
from scipy import linalg
import multiprocessing as mp
import time

def liouvillian(omega, g, gamma, N):
    """Liouvillian for the coupled system of qubit and TLS"""

    H0_q = omega * 0.5 * (-qutip.operators.sigmaz() + qutip.qeye(2))
    # drive qubit Hamiltonian
    H1_q = -0.2 * qutip.operators.sigmax()

    # drift TLS Hamiltonian
    H0_T = omega * 0.5 * (-qutip.operators.sigmaz() + qutip.qeye(2))

    # Lift Hamiltonians to joint system operators
    H0 = qutip.tensor(H0_q, qutip.qeye(2)) + qutip.tensor(qutip.qeye(2), H0_T)
    H1 = qutip.tensor(H1_q, qutip.qeye(2))

    # qubit-TLS interaction
    H_int = g * (
        qutip.tensor(
            qutip.Qobj(np.array([[0, 0], [1, 0]])),
            qutip.Qobj(np.array([[0, 1], [0, 0]])),
        )
        + qutip.tensor(
            qutip.Qobj(np.array([[0, 1], [0, 0]])),
            qutip.Qobj(np.array([[0, 0], [1, 0]])),
        )
    )

    # convert Hamiltonians to QuTiP objects
    H0 = qutip.Qobj(H0 + H_int)
    H1 = qutip.Qobj(H1)

    # Define Lindblad operators
    # Cooling on TLS
    L1 = np.sqrt(gamma * (N + 1)) * qutip.tensor(
        qutip.Qobj(np.array([[0, 1], [0, 0]])), qutip.qeye(2)
    )
    # Heating on TLS
    L2 = np.sqrt(gamma * N) * qutip.tensor(qutip.Qobj([[0, 0], [1, 0]]), qutip.qeye(2))

    # convert Lindblad operators to QuTiP objects
    L1 = qutip.Qobj(L1)
    L2 = qutip.Qobj(L2)

    # generate the Liouvillian
    L0 = qutip.liouvillian(H=H0, c_ops=[L1, L2])
    L1 = qutip.liouvillian(H=H1)

    # Shift the qubit and TLS into resonance by default
    eps0 = lambda t, args: 0.000000000001
    return [L0, [L1, eps0]], H1


def trace_A(rho):
    """Partial trace over the A degrees of freedom"""
    rho_q = np.zeros(shape=(2, 2), dtype=np.complex_)
    rho_q[0, 0] = rho[0, 0] + rho[2, 2]
    rho_q[0, 1] = rho[0, 1] + rho[2, 3]
    rho_q[1, 0] = rho[1, 0] + rho[3, 2]
    rho_q[1, 1] = rho[1, 1] + rho[3, 3]
    return qutip.Qobj(rho_q)


def S(t, T):
    """Shape function for the field update"""
    return krotov.shapes.flattop(
        t, t_start=0, t_stop=T, t_rise=0.005 * T, t_fall=0.005 * T, func="sinsq"
    )


def shape_field(eps0, T):
    """Applies the shape function S(t) to the guess field"""
    eps0_shaped = lambda t, args: eps0(t, args) * S(t, T)
    return eps0_shaped


def print_qubit_error(**args):
    """Utility function writing the qubit error to screen"""
    taus = []
    for state_T in args["fw_states_T"]:
        state_q_T = trace_A(state_T)
        taus.append(state_q_T[1, 1].real)
    J_T = 1 - np.average(taus)
    print("    qubit error: %.1e" % J_T)
    return J_T


def singlequbit_opt(omega, g, gamma, N, rho_th, rho_trg, nt, lam, T):
    tlist = np.linspace(0, T, nt)

    def chis_qubit(fw_states_T, objectives, tau_vals):
        """Calculate chis for the chosen functional"""
        chis = []
        for state_i_T in fw_states_T:
            chi_i = qutip.Qobj(
                qutip.tensor(
                    qutip.Qobj(np.diag([1, 1])),
                    qutip.tensor(qutip.Qobj(np.diag([0, 1]))),
                )
            )
            chis.append(chi_i)
        return chis

    L, H1 = liouvillian(omega, g, gamma, N)
    objectives = [krotov.Objective(initial_state=rho_th, target=rho_trg, H=L)]
    pulse_options = {L[1][1]: dict(lambda_a=lam, update_shape=lambda t: S(t, T))}
    opt_result = krotov.optimize_pulses(
        objectives,
        pulse_options,
        tlist,
        iter_stop=100000,
        propagator=krotov.propagators.DensityMatrixODEPropagator(atol=1e-10, rtol=1e-8),
        chi_constructor=chis_qubit,
        info_hook=krotov.info_hooks.chain(
            krotov.info_hooks.print_debug_information, print_qubit_error
        ),
        check_convergence=krotov.convergence.Or(
            krotov.convergence.value_below("5e-3", name="J_T"),
            krotov.convergence.check_monotonic_error,
        ),
        store_all_pulses=True,
    )
    return opt_result


def fidelity(ρ, σ):
    ρ = ρ.full()
    σ = σ.full()
    return linalg.sqrtm(linalg.sqrtm(ρ) @ σ @ linalg.sqrtm(ρ)).trace() ** 2


def eigenvalues(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return eigenValues



def get_by_iter(opt_result, iter):
    for iteration, pulses in zip(opt_result.iters, opt_result.all_pulses):
        controls = [krotov.conversions.pulse_onto_tlist(pulse) for pulse in pulses]
        objectives = opt_result.objectives_with_controls(controls)
        if iteration == iter:
            dynamics = objectives[0].mesolve(
                opt_result.tlist, options=qutip.Options(atol=1e-10, rtol=1e-8)
            )
            return dynamics.states[-1]

def trace_A(rho):
    """Partial trace over the A degrees of freedom"""
    rho_q = np.zeros(shape=(2, 2), dtype=np.complex_)
    rho_q[0, 0] = rho[0, 0] + rho[2, 2]
    rho_q[0, 1] = rho[0, 1] + rho[2, 3]
    rho_q[1, 0] = rho[1, 0] + rho[3, 2]
    rho_q[1, 1] = rho[1, 1] + rho[3, 3]
    return qutip.Qobj(rho_q)
if __name__ == "__main__":
    omega = 1  # qubit level splitting
    g = 0.2*omega  # qubit-TLS coupling
    gamma = 0.05*omega  # TLS decay rate
    N=0 # inverse bath temperature
    T = 15 # final time
    nt = 1000 # number of time steps
    lambdas=np.linspace(0.005,10,100)#Number of lambdas to probe
    def paramentric_curve(lam):
        result=singlequbit_opt(omega, g, gamma, N, rho_th, rho_trg, nt, lam, T)
        result.dump(f'results_single/lambda={lam}_krotov_results')
        return result
    rho_th = qutip.Qobj(qutip.tensor(qutip.Qobj(np.diag([1,0])), qutip.Qobj(np.diag([1,0]))))
    rho_q_trg = qutip.Qobj(np.diag([1, 0]))
    rho_T_trg = qutip.Qobj(np.diag([0, 1]))
    rho_trg = qutip.tensor(rho_q_trg, rho_T_trg)
   # print("starting...")
   # N = mp.cpu_count()
   # with mp.Pool(processes=N-2) as p:
   #     results = p.map(paramentric_curve, [i for i in lambdas])
   # end = time.time()
   # print("ending....")
    for i in lambdas:
        result=paramentric_curve(i)
    

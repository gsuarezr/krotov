import qutip
import numpy as np
import matplotlib.pylab as plt
from src import krotov
from src.krotov import overlap2a as inte
import dill
from scipy.integrate import odeint

H02=dill.load(open("normal", "rb"))
He2=dill.load(open("campo","rb"))

def conversion(vector1):
    V=[[vector1[5,0],vector1[7,0],vector1[6,0],vector1[8,0]],[vector1[7,0],vector1[12,0],vector1[10,0],vector1[13,0]],[vector1[6,0],vector1[10,0],vector1[9,0],vector1[11,0]],[vector1[8,0],vector1[13,0],vector1[11,0],vector1[14,0]]]
    R=[vector1[i,0] for i in range(1,5)]
    R[1],R[2]=R[2],R[1]
    V2=V-np.outer(R,R)
    return [v for v in R],V2

def Power(vector,omega,T,nt):
    psi=vector[0]
    Energy=np.zeros(nt)
    Ergotropy=np.zeros(nt)
    for i in range (0,nt):
        vec,covar=conversion(psi[i])
        Energy[i]=1/2*(omega**2*np.real(covar[2,2])+np.real(covar[3,3])+omega**2*(np.real(vec[2]))**2+(np.real(vec[3]))**2)-omega/2
        D=np.real((1+2/omega*np.real(Energy[i])-omega*np.abs(np.real(vec[2])+1j*np.real(vec[3]))**2)**2-4*np.abs(omega*1/2*(np.real(covar[2,2])+np.real(vec[2])**2-1/(omega**2)*(np.real(covar[3,3])+np.real(vec[3])**2))+1j*(np.real(covar[2,3])+np.real(vec[2])*np.real(vec[3]))-omega/2*(np.real(vec[2])+1j/omega*np.real(vec[3]))**2)**2)
        Ergotropy[i]=np.real(Energy[i]-(np.sqrt(D)-1)/2)
    return Ergotropy[nt-1]/T

def Ratio(vector,omega,nt):
    psi=vector[0]
    Energy=np.zeros(nt)
    Ergotropy=np.zeros(nt)
    for i in range (0,nt):
        vec,covar=conversion(psi[i])
        Energy[i]=1/2*(omega**2*np.real(covar[2,2])+np.real(covar[3,3])+omega**2*(np.real(vec[2]))**2+(np.real(vec[3]))**2)-omega/2
        D=np.real((1+2/omega*np.real(Energy[i])-omega*np.abs(np.real(vec[2])+1j*np.real(vec[3]))**2)**2-4*np.abs(omega*1/2*(np.real(covar[2,2])+np.real(vec[2])**2-1/(omega**2)*(np.real(covar[3,3])+np.real(vec[3])**2))+1j*(np.real(covar[2,3])+np.real(vec[2])*np.real(vec[3]))-omega/2*(np.real(vec[2])+1j/omega*np.real(vec[3]))**2)**2)
        Ergotropy[i]=np.real(Energy[i]-(np.sqrt(D)-1)/2)
    return Ergotropy[nt-1]/Energy[nt-1]

def hamiltonian(omega, ampl0, mu,g,gamma,Nb,T):
      """Two-level-system Hamiltonian

      Args:
          omega (float): energy separation of the qubit levels
          ampl0 (float): constant amplitude of the driving field
      """
      HA = qutip.Qobj(H02(gamma,g,Nb,omega))
      H1= - qutip.Qobj(He2(mu,omega))
      def guess_control(t, args):
          return ampl0*krotov.shapes.flattop(
              t, t_start=0, t_stop=T, t_rise=0.005, func="blackman"
          )
      def guess_control2(t, args):
          return ampl0*krotov.shapes.flattop(
              t, t_start=0, t_stop=T, t_rise=0.005, func="blackman"
          )
      
      return [HA, [H1, guess_control]]

def plot_iterations(opt_result,t,nt):
    """Plot the control fields in population dynamics over all iterations.

    This depends on ``store_all_pulses=True`` in the call to
    `optimize_pulses`.
    """

    fig, [ax_ctr,ax] = plt.subplots(nrows=2, figsize=(4, 5))
    n_iters = len(opt_result.iters)
    EEnergy=np.zeros(nt)
    for (iteration, pulses) in zip(opt_result.iters, opt_result.all_pulses):
        controls = [
            krotov.conversions.pulse_onto_tlist(pulse)
            for pulse in pulses
        ]
        objectives = opt_result.objectives_with_controls(controls)
        dynamics = objectives[0].mesolve(
            opt_result.tlist, e_ops=[]
        )
        if iteration == 0:
            ls = '--'  # dashed
            alpha = 1  # full opacity
            ctr_label = 'guess'
            pop_labels = ['0 (guess)', '1 (guess)']
        elif iteration == opt_result.iters[-1]:
            ls = '-'  # solid
            alpha = 1  # full opacity
            ctr_label = 'optimized'
            pop_labels = ['0 (optimized)', '1 (optimized)']
        else:
            ls = '-'  # solid
            alpha = 0.5 * float(iteration) / float(n_iters)  # max 50%
            ctr_label = None
            pop_labels = [None, None]
        ax_ctr.plot(
            dynamics.times,
            controls[0],
            label=ctr_label,
            color='black',
            ls=ls,
            alpha=alpha,
        )
    EField=np.transpose(np.array(opt_result.optimized_controls))
    EEnergy[0]=(np.square(EField[0]))*(t[-1]/nt)
    a=0
    for i in range (1,nt):
      a+=np.square(EField[i-1])
      EEnergy[i]=(np.square(EField[i])+a)*(t[-1]/nt)
      
    
    ax.plot(t,np.transpose(EEnergy))
    plt.legend()
    plt.show(fig)  

def get_result(omega,ampl0,mu,g,gamma,Nb,tlist,nt,stop=10000,rate=1,energy=2):
    T=tlist[-1]
    H = hamiltonian(omega,ampl0,mu,g,gamma,Nb,T)
    def S(t):
        return krotov.shapes.flattop(
        t, t_start=0, t_stop=T, t_rise=0.05 * T, t_fall=0.05 * T, func='sinsq'
         )
    r=1.5,
    r2=np.pi/8,
    theta=np.pi/4  
    pulse_options = { H[1][1]: dict(lambda_a=rate, update_shape=S)}
    r2=0.5
    theta=np.pi/4
    x=0.05
    target=qutip.Qobj(np.array([1,0,np.sqrt(x),0,np.sqrt(x),1/2*np.cosh(2*r2),0,-1/2*np.sinh(2*r2)*np.cos(theta),-1/2*np.sinh(2*r2)*np.sin(theta),1/2*np.cosh(2*r2)+x,-1/2*np.sinh(2*r2)*np.sin(theta),1/2*np.sinh(2*r2)*np.cos(theta)+x,1/2*np.cosh(2*r2),0,1/2*np.cosh(2*r2)+x]) )
    r2=1.5
    theta=np.pi/4
    initial_state=np.array([1,0,0,0,0,1/2*np.cosh(2*r2),0,-1/2*np.sinh(2*r2)*np.cos(theta),-1/2*np.sinh(2*r2)*np.sin(theta),1/2*np.cosh(2*r2),-1/2*np.sinh(2*r2)*np.sin(theta),1/2*np.sinh(2*r2)*np.cos(theta),1/2*np.cosh(2*r2),0,1/2*np.cosh(2*r2)]) 
    #initial_state=np.array([1,0,0,0,0,1/2,0,0,0,1/2,0,0,1/2,0,1/2])
    objectives = [
    krotov.Objective(
        initial_state=qutip.Qobj(initial_state), target=qutip.Qobj(target), H=H
        #initial_state=qutip.Qobj(np.array([1,0,0,0,0,1/2,0,0,0,1/2,0,0,1/2,0,1/2])), target=qutip.Qobj(target), H=H
      )]
    opt_result = krotov.optimize_pulses(
      objectives,
      pulse_options=pulse_options,
      tlist=tlist,
      gamma=(gamma,omega,Nb),
      fieldcoupling=mu,
      propagator=krotov.propagators.expm,
      chi_constructor=krotov.functionals.chis_ss,
      info_hook=krotov.info_hooks.print_table(J_T=krotov.functionals.J_T_ss),
      check_convergence=krotov.convergence.Or(
          krotov.convergence.value_below('9e-5', name='J_T'),
          krotov.convergence.check_monotonic_error,
          #krotov.convergence.check_monotonic_fidelity,

      ),
        iter_stop=stop,
        store_all_pulses=True,
        overlap=inte,norm=lambda x: 1
    )
    nt=len(tlist)
    plot_iterations(opt_result,tlist,nt)    
    power=Power(opt_result.states,omega,T,nt)
    ratio=Ratio(opt_result.states,omega,nt)
    Energy=np.zeros(nt)
    Ergotropy=np.zeros(nt)
    c01=[]
    c02=[]
    c03=[]
    a=[]
    b=[]
    c=[]
    d=[]
    for i in range (0,nt):
        vec,covar=conversion(opt_result.states[0][i])
        a.append(vec[0])
        b.append(vec[1])
        c.append(vec[2])
        d.append(vec[3])
        c01.append(covar[0,1])
        c02.append(covar[0,2])
        Energy[i]=1/2*(omega**2*np.real(covar[2,2])+np.real(covar[3,3])+omega**2*(np.real(vec[2]))**2+(np.real(vec[3]))**2)-omega/2
        D=np.real((1+2/omega*np.real(Energy[i])-omega*np.abs(np.real(vec[2])+1j*np.real(vec[3]))**2)**2-4*np.abs(omega*1/2*(np.real(covar[2,2])+np.real(vec[2])**2-1/(omega**2)*(np.real(covar[3,3])+np.real(vec[3])**2))+1j*(np.real(covar[2,3])+np.real(vec[2])*np.real(vec[3]))-omega/2*(np.real(vec[2])+1j/omega*np.real(vec[3]))**2)**2)
        Ergotropy[i]=np.real(Energy[i]-(np.sqrt(D)-1)/2)    
    return ratio,power,opt_result,Energy,Ergotropy,c01,c02,a,b,c,d

def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
        return z

def pend(y, t, F, w0, N, gamma, g):

    dydt = [-1j*(g*y[1]+F)-gamma/2*y[0], -1j*g*y[0], 1j*(g*(y[4]-y[3])-F*np.conjugate(y[1]))-gamma/2*y[2],2*g*np.imag(y[2]), -2*np.imag(g*y[2]+F*y[0])-gamma*y[4]+gamma*N,-2*1j*(g*y[6]+F*y[0])-gamma*y[5],-1j*(g*(y[5]+y[7])+F*y[1])-gamma/2*y[6],-2*1j*g*y[6]]

    return dydt


def giovanetti(w0,g,gamma,F,N,T=31):
    y0 = [0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    t = np.linspace(0, T, 10000)
    Energy2=np.zeros(len(t))
    Ergotropy2=np.zeros(len(t))
    D=np.zeros(len(t))
    sol = odeintz(pend, y0, t, args=(F, w0, N, gamma, g))
    D=(1+2*sol[:,3]-2*np.absolute(sol[:,1])**2)**2-4*np.absolute(sol[:,7]-sol[:,1]**2)**2
    Energy2=w0*sol[:,3]
    Ergotropy2=w0*(sol[:,3]-1/2*(np.sqrt(D)-1))
    return t,Energy2,Ergotropy2

def field_energy(opt_result,tlist,T,nt):
    EField=np.transpose(np.array(opt_result.optimized_controls))
    EEnergy=np.zeros(len(tlist))
    EEnergy[0]=(np.square(EField[0]))*(T/nt)
    a=0
    for i in range (1,nt):
        a+=np.square(EField[i-1])
        EEnergy[i]=(np.square(EField[i])+a)*(T/nt)
    return EEnergy[nt-1]


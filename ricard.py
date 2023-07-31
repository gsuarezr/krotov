from scipy import integrate
import scipy as scipy
import numpy as np
import matplotlib.pylab as plt
import math
import qutip
from numba import jit


@jit(nopython=True, parallel=True)
def lims2(s1,t2,t1):
    return[0,s1]
def limt2(t1):
    return [0, t1]
def limt1(t1):
    return [0, t1]
def complexintegral1(func, lim, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = integrate.nquad(real_func, lim, **kwargs)
    imag_integral = integrate.nquad(imag_func, lim, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0])

def complexintegral2(func, lim, **kwargs):
    def real_func(x,y):
        return np.real(func(x,y))
    def imag_func(x,y):
        return np.imag(func(x,y))
    real_integral = integrate.nquad(real_func, lim, **kwargs)
    imag_integral = integrate.nquad(imag_func, lim, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0])

def complexintegral3(func, lim, **kwargs):
    def real_func(x,y,z):
        return np.real(func(x,y,z))
    def imag_func(x,y,z):
        return np.imag(func(x,y,z))
    real_integral = integrate.nquad(real_func, lim, **kwargs)
    imag_integral = integrate.nquad(imag_func, lim, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0])

def complexintegral4(func, lim, **kwargs):
    def real_func(x,y,z,t):
        return np.real(func(x,y,z,t))
    def imag_func(x,y,z,t):
        return np.imag(func(x,y,z,t))
    real_integral = integrate.nquad(real_func, lim, **kwargs)
    imag_integral = integrate.nquad(imag_func, lim, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0])

def complexintegral6(func, lim, **kwargs):
    def real_func(x,y,z,t,a,b,c):
        return np.real(func(x,y,z,a,b,c))
    def imag_func(x,y,z,t,a,b,c):
        return np.imag(func(x,y,z,a,b,c))
    real_integral = integrate.nquad(real_func, lim, **kwargs)
    imag_integral = integrate.nquad(imag_func, lim, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0])

def complexintegral7(func, lim, **kwargs):
    def real_func(x,y,z,t,a,b,c):
        return np.real(func(x,y,z,t,a,b,c))
    def imag_func(x,y,z,t,a,b,c):
        return np.imag(func(x,y,z,t,a,b,c))
    real_integral = integrate.nquad(real_func, lim, **kwargs)
    imag_integral = integrate.nquad(imag_func, lim, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0])


func11 = lambda  w1,w2 : kappa**2*teta**2*4*math.pi*w1**2*4*math.pi*w2**2*np.sqrt(w1)*np.sqrt(w2)*(np.exp(-1j*w1*T[i])*np.exp(-1/2*((w1-wa)/a[l])**2-1/2*((w2-wa)/a[l])**2)*(np.exp((gamma+1j*wa)*T[i])*(-w1+w2)+np.exp(1j*(w2)*T[i])*(1j*gamma+w1-wa)+np.exp(1j*(w1)*T[i])*(-1j*gamma-w2+wa)))/(2*math.pi*(w1-w2)*(1j*gamma+w1-wa)*(1j*gamma+w2-wa)) if (w1-w2)!=0 else 0+0j

func12 = lambda  w1,w2 : kappa**2*teta**2*4*math.pi*w1**2*4*math.pi*w2**2*np.sqrt(w1)*np.sqrt(w2)*(np.exp(-1j*w1*T[i])*np.exp(-1/2*((w1-wa)/a[l])**2-1/2*((w2-wa)/a[l])**2)*(np.exp((gamma+1j*wa)*T[i])*(-w1+w2)+np.exp(1j*(w2)*T[i])*(1j*gamma+w1-wa)+np.exp(1j*(w1)*T[i])*(-1j*gamma-w2+wa)))/(2*math.pi*(w1-w2)*(1j*gamma+w1-wa)*(1j*gamma+w2-wa)) if (w1-w2)!=0 else 0+0j

func13 = lambda  w1,w2 : kappa**2*teta**2*4*math.pi*w1**2*4*math.pi*w2**2*np.sqrt(w1)*np.sqrt(w2)*1j*1/math.pi*(np.exp(-1/2*((w1-wa)/a[l])**2-1/2*((w2-wa)/a[l])**2))*((np.exp(T[i]*(2*gamma+1j*(w1-w2))))/((2*1j*gamma-w1+w2)*(gamma-1j*(w2-wa)))-2*gamma/((w1-w2)*(-2*1j*gamma+w1-w2)*(-1j*gamma+w1-wa))+(1j*np.exp(1j*T[i]*(w1-w2)))/((w1-w2)*(-1j*gamma+w2-wa))+(2*gamma*np.exp(T[i]*(gamma+1j*(w1-wa))))/((gamma+1j*(w1-wa))*(gamma-1j*(w2-wa))*(-1j*gamma+w2-wa))) if (w1-w2)!=0 else 0+0j

func21 = lambda  w: kappa*4*math.pi*w**2*np.sqrt(w)*np.exp(-gamma*T[i])*np.exp(-1/2*((w-wa)/a[l])**2)*(np.exp(gamma*T[i])-np.exp(1j*T[i]*(w-wa)))/(np.sqrt(2*math.pi)*(gamma-1j*(w-wa)))

func22 = lambda  w1,w2,w3 : kappa**3*teta**3*4*math.pi*w1**2*4*math.pi*w2**2*4*math.pi*w3**2*np.exp(-1/2*((w1-wa)/a[l])**2-1/2*((w2-wa)/a[l])**2-1/2*((w3-wa)/a[l])**2)*np.exp(-T[i]*(gamma+1j*(w2+wa)))*\
(np.exp(1j*T[i]*w3)-np.exp(T[i]*(gamma+1j*wa)))*\
np.sqrt(w1*w2*w3)*(np.exp(T[i]*(gamma+1j*(w1+w2-wa)))*(w1-w2)+np.exp(1j*T[i]*w2)*\
                   (-1j*gamma+w2-wa)+np.exp(1j*T[i]*w1)*(1j*gamma-w1+wa))/(2*np.sqrt(2)*np.sqrt(math.pi)*(math.pi)*(w1-w2)*(gamma-1j*(w3-wa))*(-1j*gamma+w1-wa)*(-1j*gamma+w2-wa)) if (w1-w2)!=0 else 0+0j

func23 = lambda  w1 : kappa*teta*4*math.pi*w1**2*(np.exp(-((w1-wa)**2/(2*a[l]**2))-1j*T[i]*wa)*np.sqrt(2/math.pi)*np.sqrt(w1)*(-np.exp(1j*T[i]*wa)*gamma+np.exp(1j*T[i]*w1)*(gamma*math.cosh(gamma*T[i])-1j*(w1-wa)*math.sinh(gamma*T[i])))/((gamma**2+(w1-wa)**2)))

func31 = lambda  w1 : kappa*teta*4*math.pi*w1**2*(np.exp(-T[i]*(gamma+1j*(w1-wa))-(w1-wa)**2/(2*a[l]**2))*(-1+np.exp(T[i]*(gamma+1j*(w1-wa))))*np.sqrt(w1))/(np.sqrt(2*math.pi)*(gamma+1j*(w1-wa)))


NoP = lambda w1 : teta**2*4*math.pi*w1**2*np.exp(-((w1-wa)/a[l])**2)

wa=1
gamma=0.05
kappa=0.035
teta = math.pi/2

T = np.linspace(0, 20, 100)
a = np.linspace(0.1, 1, 6)

NumberOfPhotons = np.zeros(len(a), dtype=complex)
ent, ay =plt.subplots()

for l in range(len(a)):
    NumberOfPhotons[l] = 1 / ((np.sqrt(2 *math.pi)) * 2) * complexintegral1(NoP, [[0, np.inf]])
    print('Number of photons =',NumberOfPhotons[l])
    Entanglement = np.zeros(len(T))
    rho = np.zeros((2, 2), dtype=complex)

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='3d'))

    #ax.axis('square') # to get a nice circular plot

    b = qutip.Bloch(fig=fig,axes=ax)
    ax.set_title('For N=' + str(np.real(round(NumberOfPhotons[l], 4))), y=1.1, fontsize=15)
    print("l=", l)
    for i in range(len(T)):
        rho11 = np.exp(-2  *gamma*  T[i]) - 2  np.exp(-2  *gamma * T[i])  np.real(
            complexintegral2(func11, [[0, np.inf], [0, np.inf]])) + \
                np.exp(-2  *gamma  *T[i])  np.absolute(complexintegral2(func12, [[0, np.inf], [0, np.inf]])) * 2 + \
                np.exp(-2 * gamma*  T[i]) * np.real(complexintegral2(func13, [[0, np.inf], [0, np.inf]]))

        rho12 = -1j  np.exp(-gamma*  T[i]) * complexintegral1(func21, [[0, np.inf]]) + \
                1j  np.exp(-gamma  *T[i]) * complexintegral3(func22, [[0, np.inf], [0, np.inf], [0, np.inf]]) + \
                1j  np.exp(-gamma * T[i]) * np.real(complexintegral1(func23, [[0, np.inf]]))

        rho22 = np.absolute(complexintegral1(func31, [[0, np.inf]])) * 2 + 1 - np.exp(-2*  gamma * T[i])

        rho[0, 0] = rho22
        rho[0, 1] = np.conjugate(rho12)
        rho[1, 0] = rho12
        rho[1, 1] = rho11
        N = 1 / (np.trace(rho))
        rho = N * rho
        Entanglement[i] = 1 - np.trace(np.dot(rho, rho))
        print("i=", i)
        print(Entanglement[i])
        point1 = [2  np.real(rho[0, 1]), -2  np.imag(rho[0, 1]), rho[0, 0] - rho[1, 1]]
        b.add_points(point1)
    b.render(fig=fig, axes=ax)
    fig.savefig("BlochN=" + str(np.real(round(NumberOfPhotons[l], 4))) + "(New).png", format="png", dpi=150)
    ay.plot(T, Entanglement, label="N =" + str(np.real(round(NumberOfPhotons[l], 4))))

ay.set_xlabel("Time")
ay.set_ylabel("Entanglement")
ay.legend(loc='upper center', bbox_to_anchor=(1.2, 1), shadow=True, ncol=1)
ent.savefig("EntanglementvsT(New).png", format="png", dpi=150, bbox_inches="tight")
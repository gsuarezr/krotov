from numpy import array,vectorize,real,flip,loadtxt,block
import sympy as sp
from sympy import Function,Symbol,symbols,zeros,Matrix,sqrt,simplify,solve,diff,dsolve,lambdify
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
import krotov


def field_discrete(lista,t,dt):
    return lista[int(t/dt)]

def derivative(lista,t,dt):
    return array([(lista[j]-lista[j-1])/(dt) if (j!=0)&(j!=len(lista)) else lista[j] for j in range(0,len(lista))])[int(t/dt)]

guess_field=vectorize(krotov.shapes.flattop)


def vectorfieldcmReales( t,w, p):
    """
    Defines the differential equations for  system.

    Arguments:
        w :  vector of the state variables:
                  w = [v11,v12,v13,v14,v22,v23,v24]
        t :  time
        p: vector of parameters
            p=[g,field,dt,mu]
    """
    g,field,dt,mu,wsol,eqs=p
    x1_diff,x2_diff,v11_diff,v12_diff,v13_diff,v14_diff,v22_diff,v23_diff,v24_diff=eqs
    v11,v12,v13,v14,v22,v23,v24 = w
    # Create f = (x1',y1',x2',y2'):
    f = []
    f.append(v11_diff(field_discrete(wsol.y[0],t,dt),field_discrete(wsol.y[1],t,dt),g,derivative(field,t,dt),field_discrete(field,t,dt),mu,v11))
    f.append(v12_diff(field_discrete(wsol.y[0],t,dt),field_discrete(wsol.y[1],t,dt),g,derivative(field,t,dt),field_discrete(field,t,dt),mu,v12))
    f.append(v13_diff(field_discrete(wsol.y[0],t,dt),field_discrete(wsol.y[1],t,dt),g+0j,derivative(field,t,dt),field_discrete(field,t,dt),mu,v13,t))
    f.append(v14_diff(field_discrete(wsol.y[0],t,dt),field_discrete(wsol.y[1],t,dt),g+0j,derivative(field,t,dt),field_discrete(field,t,dt),mu,v14,t))
    f.append(v22_diff(field_discrete(wsol.y[0],t,dt),field_discrete(wsol.y[1],t,dt),g,derivative(field,t,dt),field_discrete(field,t,dt),mu,v22))
    f.append(v23_diff(field_discrete(wsol.y[0],t,dt),field_discrete(wsol.y[1],t,dt),g+0j,derivative(field,t,dt),field_discrete(field,t,dt),mu,v23,t))
    f.append(v24_diff(field_discrete(wsol.y[0],t,dt),field_discrete(wsol.y[1],t,dt),g+0j,derivative(field,t,dt),field_discrete(field,t,dt),mu,v24,t))

    return real(f)

def get_diff(p,initial_vals,file,func,eqs,xs=0,backwards=False,from_txt=False):
    """
    Arguments:
        p: vector of parameters
            p=[ini,fin,lt,g,mu,wsol]
    """
    wsol=xs
    time,g,mu=p
    ini=time[0]
    fin=time[-1]
    lt=len(time)
    if from_txt:
        field=loadtxt(file)
    else:
        field=file
    dt=fin/(lt-1)
    p2=[g,field,dt,mu,wsol,eqs]
    ranged=[ini,fin]
    if backwards:
        time=flip(time)
        ranged=[fin,ini]

    wsol2 = ivp(func, ranged,initial_vals, args=(p2,),t_eval=time)
    return wsol2



def vectorfield( t,w, p):
    """
    Defines the differential equations for  system.

    Arguments:
        w :  vector of the state variables:
                  w = [x1,x2]
        t :  time
        p: vector of parameters
            p=[g,field,dt,mu]
    """
    x1, x2 = w
    g,field,dt,mu,wsol,eqs=p
    x1_diff,x2_diff,v11_diff,v12_diff,v13_diff,v14_diff,v22_diff,v23_diff,v24_diff=eqs
    # Create f = (x1',y1',x2',y2'):
    f = [x1_diff(x1,x2,g,derivative(field,t,dt),field_discrete(field,t,dt),mu),x2_diff(x1,x2,g,derivative(field,t,dt),field_discrete(field,t,dt),mu)]
    return f

def get_plot(wsol,func):
    plt.figure(figsize=(18, 6), dpi=80)
    if func==vectorfield:
        plt.plot(wsol.t,wsol.y[0],label=r'$x_{1}$')
        plt.plot(wsol.t,wsol.y[1],'-.',label=r'$x_{2}$')
    if func==vectorfieldcmReales:
        plt.plot(wsol.t,wsol.y[0],label=r'$v_{11}$')
        plt.plot(wsol.t,wsol.y[1],label=r'$v_{12}$')
        plt.plot(wsol.t,wsol.y[2],label=r'$v_{13}$')
        plt.plot(wsol.t,wsol.y[3],label=r'$v_{14}$')
        plt.plot(wsol.t,wsol.y[4],label=r'$v_{22}$')
        plt.plot(wsol.t,wsol.y[5],label=r'$v_{23}$')
        plt.plot(wsol.t,wsol.y[6],label=r'$v_{24}$')
    plt.legend()
    return plt.show()


def get_VFinal(p1,p2):
    t=Symbol('t')
    v11 = Function('v11')(t)
    v12 = Function('v12')(t)
    v13= Function('v13')(t)
    v14 = Function('v14')(t)
    v22= Function('v22')(t)
    v23 = Function('v23')(t)
    v24= Function('v24')(t)
    v33 = Function('v33')(t)
    v34= Function('v34')(t)
    v44 = Function('v44')(t)
    x1 = Function('x1')(t)
    x2 = Function('x2')(t)
    p1= Function('p1')(t)
    p2 = Function('p2')(t)
    g= symbols('g')
    e=Function('e')(t)
    mu=symbols('mu')
    F=zeros(2,2)
    F[0,1]=1
    F[1,0]=-1
    B=zeros(2,2)
    B[0,0]=1
    B[1,1]=-1
    A=block([
            [g*B+F,             Matrix([[0,0],[0,0]])],
            [Matrix([[0,0],[0,0]]), g*B  +F             ]
            ])
    v=[]
    v.append(x1-sqrt(2)*mu*e/(1-pow(g,2)))
    v.append(x2+sqrt(2)*mu*e*g/(1-pow(g,2)))
    v.append(p1)
    v.append(p2)
    v=Matrix(v)
    r=simplify(diff(v,t)-Matrix(A)*Matrix(v))
    r2=r[-2:]
    sol=solve([r[0],r[1]],diff(x1,t),diff(x2,t))
    d1=simplify(sol[diff(x1,t)])
    d2=simplify(sol[diff(x2,t)])
    x1_diff=lambdify((x1,x2,g,diff(e,t),e,mu),d1,"numpy")
    x2_diff=lambdify((x1,x2,g,diff(e,t),e,mu),d2,"numpy")
    alpha=sqrt(2)*mu*e/(1-pow(g,2))
    beta=sqrt(2)*mu*e*g/(1-pow(g,2))
    F=zeros(2,2)
    F[0,1]=1
    F[1,0]=-1
    B=zeros(2,2)
    B[0,0]=1
    B[1,1]=-1
    A=block([
            [g*B+F,             Matrix([[0,0],[0,0]])],
            [Matrix([[0,0],[0,0]]), g*B  +F             ]
            ])
    V=Matrix([[v11-2*x1+alpha**2,v12-alpha*x2+beta*x1-alpha*beta,v13-alpha*p1,v14-alpha*p2],[v12-alpha*x2+beta*x1-alpha*beta,v22+beta**2+2*x2,v23+beta*p1,v24+beta*p2],[v13-alpha*p1,v23+beta*p1,v33,v34],[v14-alpha*p2,v24+beta*p2,v34,v44]])
    r=simplify(diff(V,t)-Matrix(A)*Matrix(V)-Matrix(V)*Matrix(A).T)
    dV=diff(V,t)
    VFinal=dV.subs({diff(x1,t):d1,diff(x2,t):d2,diff(p1,t):diff(dsolve(r2)[0].rhs,t),diff(p2,t):diff(dsolve(r2)[1].rhs,t),p1:dsolve(r2)[0].rhs,p2:dsolve(r2)[1].rhs,"C1":c1(p1,p2,g),"C2":c2(p1,p2,g)})
    v11_diff=lambdify((x1,x2,g,diff(e,t),e,mu,v11),solve(VFinal[0,0],diff(v11,t))[0],"numpy")
    v12_diff=lambdify((x1,x2,g,diff(e,t),e,mu,v12),solve(VFinal[0,1],diff(v12,t))[0],"numpy")
    v13_diff=lambdify((x1,x2,g,diff(e,t),e,mu,v13,t),solve(VFinal[0,2],diff(v13,t))[0],"numpy")
    v14_diff=lambdify((x1,x2,g,diff(e,t),e,mu,v14,t),solve(VFinal[0,3],diff(v14,t))[0],"numpy")
    v22_diff=lambdify((x1,x2,g,diff(e,t),e,mu,v22),solve(VFinal[1,1],diff(v22,t))[0],"numpy")
    v23_diff=lambdify((x1,x2,g,diff(e,t),e,mu,v23,t),solve(VFinal[1,2],diff(v23,t))[0],"numpy")
    v24_diff=lambdify((x1,x2,g,diff(e,t),e,mu,v24,t),solve(VFinal[1,3],diff(v24,t))[0],"numpy")
    p_1=lambdify((p1,p2,g,t),(dsolve(r2)[0].rhs).subs({"C1":c1(p1,p2,g),"C2":c2(p1,p2,g)}))
    p_2=lambdify((p1,p2,g,t),(dsolve(r2)[1].rhs).subs({"C1":c1(p1,p2,g),"C2":c2(p1,p2,g)}))
    return [(x1_diff,x2_diff,v11_diff,v12_diff,v13_diff,v14_diff,v22_diff,v23_diff,v24_diff),(p_1,p_2)]


def sol_vector(r,p,field):
    eqs=get_VFinal(r[2],r[3])
    wsol=get_diff(p,[r[0],r[1]],field,vectorfield,eqs[0])
    return eqs,wsol

def c2(p1,p2,g):
    num=-g*p1+p1*sqrt(g**2 -1) - p2
    dem=2*sqrt(g**2 -1)
    return num/dem

def c1(p1,p2,g):
    num=g*p1+p1*sqrt(g**2 -1) + p2
    dem=2*sqrt(g**2 -1)
    return num/dem
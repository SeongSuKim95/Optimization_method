import numpy as np
import argparse

def Steepest_Gradient_method(Q,b,initial_x,n):
    xs = []
    gs = []
    alphas = []

    x0 = np.array(initial_x)
    xs.append(x0)

    g0 = np.dot(Q,x0) - b
    gs.append(g0)

    alpha0 = g0.dot(g0) / g0.dot(Q).dot(g0)
    alphas.append(alpha0)

    for i in range(n):

        x = xs[i] - alphas[i] * gs[i]
        xs.append(x)
        if np.all(x < 1e-3) :
            break
        g = np.dot(Q,xs[i+1]) - b
        gs.append(g)

        alpha = g.dot(g)/g.dot(Q).dot(g)
        alphas.append(alpha)

    xs = np.round(xs, 3)
    alphas = np.round(alphas, 3)

    for j in range(xs.shape[0]):
        print(f"Iteration {j}")
        print(f"Value of x : {xs[j]}")
        print(f"Value of a : {alphas[j]}")

    return xs

def Newtons_method(Q,b,initial_x):
    x = np.array(initial_x)
    g = np.dot(Q,x) - b
    F = Q
    x = x - np.dot(np.linalg.inv(F),g)
    print(f"Solution of x : {x}")
    return x

def Conjugate_Gradient_method(Q,b,initial_x):
    n = b.shape[0]
    xs = []
    gs = []
    ds = []
    alphas = []

    x0 = np.array(initial_x)
    # x0 = np.random.rand(n)
    xs.append(x0)

    g0 = np.dot(Q,x0) - b
    # Original :r0 = b - np.dot(Q,x0) # g(k)
    gs.append(g0)

    d0 = -g0
    # Original p0 = r0 # d(k) = -g(k)
    ds.append(d0)

    alpha0 = - g0.dot(d0) / d0.dot(Q).dot(d0)
    # Original alpha0 = p0.dot(p0)/p0.dot(Q).dot(p0) # a(k) = - g(k)T * d(k) / d(k)T * A * d(k)
    alphas.append(alpha0)

    for i in range(n):

        x = xs[i] + alphas[i] * ds[i]
        xs.append(x)
        g = np.dot(Q,xs[i+1]) - b
        #g = gs[i] - alphas[i] * Q.dot(ds[i]) # g(k+1) = gradient f(x(k+1))
        gs.append(g)

        beta = (gs[i+1]).dot(Q).dot(ds[i])/(ds[i]).dot(Q).dot(ds[i]) # beta = -g(k)T*d(k)/d(k)T*Q*d(k)

        d = -g + beta *ds[i]
        ds.append(d)

        alpha = -gs[i+1].dot(ds[i+1])/(ds[i+1]).dot(Q).dot(ds[i+1])
        alphas.append(alpha)

    xs = np.round(xs, 3)
    alphas = np.round(alphas, 3)
    ds = np.round(ds, 3)

    for j in range(xs.shape[0]):
        print(f"Iteration {j}")
        print(f"Value of x : {xs[j]}")
        print(f"Value of a : {alphas[j]}")
        print(f"Value of d : {ds[j]}")

    return xs

def Quasi_Newtons_method(Q,b,initial_x,n):

    xs = []
    Hs = []
    gs = []
    ds = []
    alphas = []

    x0 = np.array(initial_x)
    # x0 = np.random.rand(n)
    xs.append(x0)

    H0 = np.identity(b.shape[0])
    Hs.append(H0)

    g0 = np.dot(Q, x0) - b
    gs.append(g0)

    d0 = - H0.dot(g0)
    ds.append(d0)

    alpha0 = - g0.dot(d0) / d0.dot(Q).dot(d0)
    alphas.append(alpha0)

    for i in range(n):
        x = xs[i] + alphas[i] * ds[i]
        xs.append(x)
        del_x = alphas[i] * ds[i]

        g = np.dot(Q, xs[i + 1]) - b
        gs.append(g)

        del_g = g - gs[i]

        temp0 = del_x - H0.dot(del_g)
        temp1 = del_g.dot(temp0)
        H = Hs[i] + np.dot(temp0.reshape(temp0.shape[0],1), temp0.reshape(1,temp0.shape[0]))/temp1
        Hs.append(H)

        d = -H.dot(g)
        ds.append(d)

        alpha = - g.dot(d) / d.dot(Q).dot(d)
        alphas.append(alpha)

    xs = np.round(xs,3)
    alphas = np.round(alphas,3)
    ds = np.round(ds,3)

    for j in range(xs.shape[0]):
        print(f"Iteration {j}")
        print(f"Value of x : {xs[j]}")
        print(f"Value of a : {alphas[j]}")
        print(f"Value of d : {ds[j]}")
    return xs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Optimization')

    parser.add_argument('-method', required=True, type=str)  # 어떤 알고리즘을 쓸지
    parser.add_argument('-dimension', required=True,type = int)
    parser.add_argument('-Q', required=True,nargs = 9,type = float)  # Q matrix
    parser.add_argument('-b', required=True,nargs = 3,type = float)  # b matrix
    parser.add_argument('-x0', required=True,nargs = 3, type = float) # x0 , initial point

    args = parser.parse_args()

    Q = np.array(args.Q).reshape(args.dimension,args.dimension)
    b = np.array(args.b)
    x0 = np.array(args.x0)

    if args.method == "Steepest":
        Steepest_Gradient_method(Q,b,initial_x=x0,n=3)
    elif args.method == "Newton":
        Newtons_method(Q,b,initial_x=x0)
    elif args.method == "Conjugate":
        Conjugate_Gradient_method(Q,b,initial_x=x0)
    elif args.method == "Quasi-Newton":
        Quasi_Newtons_method(Q,b,initial_x=x0,n=3)
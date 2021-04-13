import numpy as np

def Conjugate_Gradient_method(Q,b):
    n = b.shape[0]
    xs = []
    rs = []
    ps = []
    alphas = []

    x0 = np.array([2,2,1])
    # x0 = np.random.rand(n)
    xs.append(x0)

    r0 = b - np.dot(Q,x0) # g(k)
    rs.append(r0)

    p0 = r0 # d(k) = -g(k)
    ps.append(p0)

    alpha0 = p0.dot(p0)/p0.dot(Q).dot(p0) # a(k) = - g(k)T * d(k) / d(k)T * A * d(k)
    alphas.append(alpha0)

    print(rs)
    print(alphas)
    print(ps)

    for i in range(n):

        r = rs[i] - alphas[i] * Q.dot(ps[i]) # g(k+1) = gradient f(x(k+1))
        rs.append(r)
        beta = np.dot(r,r)/(rs[i].dot(rs[i])) # beta = -g(k)T*d(k)/d(k)T*Q*d(k)

        alpha = ps[i].dot(rs[i])/(ps[i]).dot(Q).dot(ps[i])

        alphas.append(alpha)

        x = xs[i] + alpha * ps[i]

        xs.append(x)

        p = r + beta * ps[i]
        ps.append(p)

    return xs

if __name__ == '__main__':
    A = np.array([[3,0,1],[0,4,2],[1,2,3]])
    b = np.array([3,0,1])
    print(b.shape[0])
    c1 = np.linalg.inv(A).dot(b)
    print("the math sol is",c1)
    c2 = CG(A,b)
    print("the numerical sol is",c2)

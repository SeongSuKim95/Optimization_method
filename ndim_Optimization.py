import numpy as np
import argparse

def Steepest_Gradient_method(Q,b,initial_x,n):
    x_buffer = [] # Iteration 에 따른 x 값을 저장하기 위한 list
    g_buffer = [] # Iteration 에 따른 g 값을 저장하기 위한 list
    alpha_buffer = [] # Iteration 에 따른 alpha 값을 저장하기 위한 list

    x0 = np.array(initial_x) # Initial point
    x_buffer.append(x0)

    g0 = np.dot(Q,x0) - b # G Initial value
    g_buffer.append(g0)

    alpha0 = g0.dot(g0) / g0.dot(Q).dot(g0) # Alpha Initial value
    alpha_buffer.append(alpha0)

    for i in range(n):
        x = x_buffer[i] - alpha_buffer[i] * g_buffer[i] # x(k+1) = x(k) - a(k)g(k)
        x_buffer.append(x)
        g = np.dot(Q,x_buffer[i+1]) - b # g = Qx-b
        g_buffer.append(g)

        alpha = g.dot(g)/g.dot(Q).dot(g) # a(k) = g(k)T*g(k)/g(k)T*Q*g(k)
        alpha_buffer.append(alpha)

    x_buffer = np.round(x_buffer, 3) #
    alpha_buffer = np.round(alpha_buffer, 3)

    for j in range(x_buffer.shape[0]):
        print(f"Iteration {j}")
        print(f"Value of x : {x_buffer[j]}")
        print(f"Value of a : {alpha_buffer[j]}")

    return x_buffer

def Newtons_method(Q,b,initial_x):
    x = np.array(initial_x)
    g = np.dot(Q,x) - b # g = Qx -b
    F = Q # Hessian matrix = Q
    x = x - np.dot(np.linalg.inv(F),g) # x = x - F_inv*g
    print(f"Solution of x : {x}")
    return x

def Conjugate_Gradient_method(Q,b,initial_x):
    n = b.shape[0]
    x_buffer = []
    g_buffer = []
    d_buffer = []
    alpha_buffer = []

    x0 = np.array(initial_x)
    x_buffer.append(x0)

    g0 = np.dot(Q,x0) - b
    g_buffer.append(g0)

    d0 = -g0
    d_buffer.append(d0)

    alpha0 = - g0.dot(d0) / d0.dot(Q).dot(d0) # a(k) = - g(k)T * d(k) / d(k)T * A * d(k)
    alpha_buffer.append(alpha0)

    for i in range(n):

        x = x_buffer[i] + alpha_buffer[i] * d_buffer[i]
        x_buffer.append(x)
        g = np.dot(Q,x_buffer[i+1]) - b # g(k+1) = gradient f(x(k+1))
        g_buffer.append(g)

        beta = (g_buffer[i+1]).dot(Q).dot(d_buffer[i])/(d_buffer[i]).dot(Q).dot(d_buffer[i]) # beta = -g(k)T*d(k)/d(k)T*Q*d(k)

        d = -g + beta * d_buffer[i]
        d_buffer.append(d)

        alpha = -g_buffer[i+1].dot(d_buffer[i+1])/(d_buffer[i+1]).dot(Q).dot(d_buffer[i+1])
        alpha_buffer.append(alpha)

    x_buffer = np.round(x_buffer, 3)
    alpha_buffer = np.round(alpha_buffer, 3)
    d_buffer = np.round(d_buffer, 3)

    for j in range(x_buffer.shape[0]):
        print(f"Iteration {j}")
        print(f"Value of x : {x_buffer[j]}")
        print(f"Value of a : {alpha_buffer[j]}")
        print(f"Value of d : {d_buffer[j]}")

    return x_buffer

def Quasi_Newtons_method(Q,b,initial_x,n):

    x_buffer = [] # Iteration 에 따른 x 값을 저장하기 위한 list
    H_buffer = [] # Iteration 에 따른 H 값을 저장하기 위한 list
    g_buffer = [] # Iteration 에 따른 g 값을 저장하기 위한 list
    d_buffer = [] # Iteration 에 따른 d 값을 저장하기 위한 list
    alpha_buffer = [] # Iteration 에 따른 alpha 값을 저장하기 위한 list

    x0 = np.array(initial_x) # Initial point
    x_buffer.append(x0)

    H0 = np.identity(b.shape[0]) # Initial value of H0 : Identity matrix
    H_buffer.append(H0)

    g0 = np.dot(Q, x0) - b # G Initial value
    g_buffer.append(g0)

    d0 = - H0.dot(g0)
    d_buffer.append(d0)

    alpha0 = - g0.dot(d0) / d0.dot(Q).dot(d0) # Alpha Initial value
    alpha_buffer.append(alpha0)

    for i in range(n):
        x = x_buffer[i] + alpha_buffer[i] * d_buffer[i]
        x_buffer.append(x)
        del_x = alpha_buffer[i] * d_buffer[i] # x[k+1]- x[k]

        g = np.dot(Q, x_buffer[i + 1]) - b
        g_buffer.append(g)

        del_g = g - g_buffer[i] # g[k+1] - g[k]

        temp0 = del_x - H0.dot(del_g)
        temp1 = del_g.dot(temp0)
        H = H_buffer[i] + np.dot(temp0.reshape(temp0.shape[0], 1), temp0.reshape(1, temp0.shape[0])) / temp1
        # H1 = H0 + (del_x - h * del_g)(del_x - h * del_g)T/del_g * ( del_x - h * del_g)
        H_buffer.append(H)

        d = -H.dot(g) # d = -H*g
        d_buffer.append(d)

        alpha = - g.dot(d) / d.dot(Q).dot(d) # a = -gT*d/dT*Q*d
        alpha_buffer.append(alpha)

    x_buffer = np.round(x_buffer, 3)
    alpha_buffer = np.round(alpha_buffer, 3)
    d_buffer = np.round(d_buffer, 3)

    for j in range(x_buffer.shape[0]):
        print(f"Iteration {j}")
        print(f"Value of x : {x_buffer[j]}")
        print(f"Value of a : {alpha_buffer[j]}")
        print(f"Value of d : {d_buffer[j]}")

    return x_buffer

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
import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

def function(t: float, y: float):
    return t - (y**2)

# Question 1 - Euler Method
def modified_eulers():
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10
    # set h
    h = (end_of_t - start_of_t) / num_of_iterations
    t = np.zeros(num_of_iterations+1)
    w = np.zeros(num_of_iterations+1)
    #  set original w
    w[0] = 1

    for cur_iteration in range(0, num_of_iterations):
        incremented_function_call = function(t[cur_iteration], w[cur_iteration])
        w[cur_iteration + 1] = w[cur_iteration] + h * incremented_function_call
        t[cur_iteration+1] = t[cur_iteration] + h

    rounded_w = w[cur_iteration + 1]
    print("%.5f" % rounded_w)

    return None

# Question 2 - Runge-Kutta 
def midpoint_method():
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10
    # set h
    h = (end_of_t - start_of_t) / num_of_iterations
    t = np.zeros(num_of_iterations+1)
    w = np.zeros(num_of_iterations+1)
    #  set original w
    w[0] = 1

    for cur_iteration in range(0, num_of_iterations): 
        # get the k's
        k1 = h * function(t[cur_iteration], w[cur_iteration])
        k2 = h * function(t[cur_iteration] + h/2, w[cur_iteration] + k1/2)
        k3 = h * function(t[cur_iteration] + h/2, w[cur_iteration] + k2/2)
        k4 = h * function(t[cur_iteration] + h, w[cur_iteration] + k3) 

        w[cur_iteration + 1] = w[cur_iteration] + 1/6 * (k1 + (2 * k2) + (2 * k3) + k4)
        t[cur_iteration+1] = t[cur_iteration] + h
        
    rounded_w = w[cur_iteration + 1]
    print("%.5f" % rounded_w)

    return None

# Question 3 -  Gaussian elimination and backward substitution
def gauss_jordan(A):
    n = len(A)

    # Perform elimination
    for i in range(n):
        # Find pivot row
        # Divide pivot row by pivot element
        pivot = A[i, i]
        A[i, :] /= pivot
        # Eliminate entries below pivot
        for j in range(i+1, n):
            factor = A[j, i]
            A[j, :] -= factor * A[i, :] # operation 2 of row operations

    # Perform backward-substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = A[i, -1]
        for j in range(i+1, n):
            x[i] -= A[i, j] * x[j]
    
    return x

# Question 4 - LU Factorization
def get_determinant(B):
    # print determinant
    rounded_det = round(np.linalg.det(B))
    print("%.5f" % rounded_det)
    

def get_lu_matrix(B):
    # get current shape 
    n = B.shape[0]
    # initialize L matrix for lower-triangular
    # initialize U matrix for upper-triangular
    L = np.zeros_like(B)
    U = np.zeros_like(B)
    
    # factorization
    for curr_it in range(n):
        L[curr_it,curr_it] = 1
        U[curr_it,curr_it:] = B[curr_it,curr_it:] - L[curr_it,:curr_it] @ U[:curr_it,curr_it:]
        L[curr_it+1:,curr_it] = (B[curr_it+1:,curr_it] - L[curr_it+1:,:curr_it] @ U[:curr_it,curr_it]) / U[curr_it,curr_it]
    
    # print both matrix
    result_L =  L.astype(float)
    result_U =  U.astype(float)
    print(result_L)
    print()
    print(result_U)

    return None

# Question 5 - diagonally dominate
def check_diagonally_dominate(C):
    # for every row find sum of elements
    diagonal_coef = np.diag(np.abs(C))
    # row sum without diagonal minus the diagonal 
    row_sum = np.sum(np.abs(C), axis=1) - diagonal_coef
    # if diagonal is greater, then it is diagonally dominate
    if np.all(diagonal_coef > row_sum):
        print ("True")
    else:
        print ("False")

# Question 6 - positive definite
def check_positive_definite(D):
    # check if every eigenvalue is positive
    if np.all(np.linalg.eigvals(D) > 0):
        print ("True")
    else:
        print ("False")

if __name__ == "__main__":
    # Question 1
    modified_eulers()
    print()

    # Question 2
    midpoint_method()
    print()

    # Question 3
    A = np.array([[2, -1, 1, 6],
              [1, 3, 1, 0],
              [-1, 5, 4, -3]], dtype=float)
    x = gauss_jordan(A)
    print(x)
    print()

    # Question 4
    B = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]
              ])
    get_determinant(B)
    print()
    get_lu_matrix(B)
    print()

    # Question 5
    C = np.array([[9, 0, 5, 2, 1],
              [3, 9, 1, 2, 1],
              [0, 1, 7, 2, 3],
              [4, 2, 3, 12, 2],
              [3, 2, 4, 0, 8]
              ])
    check_diagonally_dominate(C)
    print()

    # Question 6
    D = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]
              ])
    check_positive_definite(D)
 
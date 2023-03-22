# A square matrix M is said to be:
# diagonal if the entries Outside the main-diagonal are all zeros
# Scalar: if it is a diagonal matrix, all of whose diagonal elements are equal
# identity: ff it is a scalar matrix, all of whose diagonal elements are equal to 1

# Write a function named matrix_type that accepts a matrix M as argument and returns the type of matrix. It should
# be one of these strings: diagonal1, scalar, identity, non-diagonal. The type you output should be the most
# appropriate one for the given matrix.


import itertools
def matrix_type(matrix: list):
    isDiagonal = True
    diagonal = matrix[0][0]
    for row, column in itertools.product(range(len(matrix)), range(len(matrix))):
        if(row != column and matrix[row][column] != 0):
            isDiagonal = False

    # Checking for scaler matrix
    isScaler = True if isDiagonal else False
    for index in range(len(matrix)):
        if(matrix[index][index] != diagonal):
            isScaler = False

    # Cheking for idendentity
    isIdentity = False
    if(isScaler and diagonal == 1):
        isIdentity = True

    if(isIdentity):
        print("identity")
    elif(isScaler):
        print("scaler")
    elif(isDiagonal):
        print("diagonal")
    else:
        print("non-diagonal")

# Driver code block
if __name__ == "__main__":
    matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    matrix_type(matrix)
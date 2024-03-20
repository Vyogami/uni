/*Aim: To perform strassen multiplication

Complexity: O(log(n^3))

Sample Input:

Enter rows and columns for Matrix A: 3 2
Enter Matrix A:
1 2
3 4
5 6
Enter rows and columns for Matrix B: 2 4
Enter Matrix B:
7 8 9 10
11 12 13 14


Pseudo Code:

Strassen(A, B, n)

Step 1. If n = 1, then
Step 2. Return single cell matrix multiplication: A[0][0] * B[0][0]
Step 3. Divide matrix A into 4 matrices: A11, A12, A21, A22 of size n/2 x n/2
Step 4. Divide matrix B into 4 matrices: B11, B12, B21, B22 of size n/2 x n/2
Step 5. Calculate seven products using Strassen's formulas:
1. P1 = Strassen(A11, (B12 - B22), n/2)
2. P2 = Strassen((A11 + A12), B22, n/2)
3. P3 = Strassen((A21 + A22), B11, n/2)
4. P4 = Strassen(A22, (B21 - B11), n/2)
5. P5 = Strassen((A11 + A22), (B11 + B22), n/2)
6. P6 = Strassen((A12 - A22), (B21 + B22), n/2)
7. P7 = Strassen((A11 - A21), (B11 + B12), n/2)
Step 6. Calculate the resultant matrix C:
1. C11 = P5 + P4 - P2 + P6
2. C12 = P1 + P2
3. C21 = P3 + P4
4. C22 = P5 + P1 - P3 - P7
Step 7. Return matrix C constructed from C11, C12, C21, C22


*/


#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 10

void printMatrix(const char* display, int matrix[MAX_SIZE][MAX_SIZE], int rows, int cols) {
    printf("\n%s =>\n", display);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%10d", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void addMatrix(int matrixA[MAX_SIZE][MAX_SIZE], int matrixB[MAX_SIZE][MAX_SIZE], int matrixC[MAX_SIZE][MAX_SIZE], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrixC[i][j] = matrixA[i][j] + matrixB[i][j];
        }
    }
}

void multiplyMatrix(int matrixA[MAX_SIZE][MAX_SIZE], int matrixB[MAX_SIZE][MAX_SIZE], int resultMatrix[MAX_SIZE][MAX_SIZE], int rowA, int colA, int rowB, int colB) {
    if (colA != rowB) {
        printf("\nError: The number of columns in Matrix A must be equal to the number of rows in Matrix B\n");
        return;
    }

    for (int i = 0; i < rowA; i++) {
        for (int j = 0; j < colB; j++) {
            resultMatrix[i][j] = 0;
            for (int k = 0; k < colA; k++) {
                resultMatrix[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

int main() {
    int matrixA[MAX_SIZE][MAX_SIZE], matrixB[MAX_SIZE][MAX_SIZE], resultMatrix[MAX_SIZE][MAX_SIZE];
    int rowA, colA, rowB, colB;

    printf("Enter rows and columns for Matrix A: ");
    scanf("%d %d", &rowA, &colA);

    printf("Enter Matrix A:\n");
    for (int i = 0; i < rowA; i++)
        for (int j = 0; j < colA; j++)
            scanf("%d", &matrixA[i][j]);

    printf("Enter rows and columns for Matrix B: ");
    scanf("%d %d", &rowB, &colB);

    printf("Enter Matrix B:\n");
    for (int i = 0; i < rowB; i++)
        for (int j = 0; j < colB; j++)
            scanf("%d", &matrixB[i][j]);

    printMatrix("Matrix A", matrixA, rowA, colA);
    printMatrix("Matrix B", matrixB, rowB, colB);

    multiplyMatrix(matrixA, matrixB, resultMatrix, rowA, colA, rowB, colB);

    printMatrix("Result Matrix", resultMatrix, rowA, colB);

    return 0;
}
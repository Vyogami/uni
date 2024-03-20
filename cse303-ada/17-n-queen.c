/*Aim: To perform fractional knapsack

Complexity: O(n)
> here "m" is the length of the first string and "n" is the length of the second string.

Sample Input:

Enter number of Queens: 4


Solution 1:

-  Q  -  -
-  -  -  Q
Q  -  -  -
-  -  Q  -


Solution 2:

-  -  Q  -
Q  -  -  -
-  -  -  Q
-  Q  -  -


PSEUDO CODE :

Initialize:
- board[20] as an array to represent the chessboard
- count as a variable to track the number of solutions

Function print(n):
    Increment count
    Print the current solution number

    For each row i from 1 to n:
        For each column j from 1 to n:
            If board[i] is equal to j:
                Print "Q" (Queen placed at this position)
            Else:
                Print "-" (Empty square)
        Print a new line

Function place(row, column):
    For each i from 1 to row - 1:
        If board[i] is equal to column OR
            the absolute difference between board[i] and column is equal to the absolute difference between i and row:
            Return 0 (It's not a safe position)
    Return 1 (It's a safe position)

Function queen(row, n):
    For each column from 1 to n:
        If place(row, column) returns 1 (safe to place):
            Set board[row] to column
            If row is equal to n:
                Call print(n) to print the current board configuration
            Else:
                Recursively call queen(row + 1, n)

Main Function:
    Declare n
    Prompt the user to "Enter number of Queens: "
    Read n
    Call queen(1, n)

*/

#include <stdio.h>
#include <stdlib.h>

int board[20], count;

void print(int n) {
    printf("\n\nSolution %d:\n\n", ++count);
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            if (board[i] == j)
                printf(" Q "); // Queen at i, j
            else
                printf(" - "); // Empty square
        }
        printf("\n");
    }
}

int place(int row, int column) {
    for (int i = 1; i <= row - 1; i++) {
        if (board[i] == column)
            return 0;
        else if (abs(board[i] - column) == abs(i - row))
            return 0;
    }

    return 1;
}

void queen(int row, int n) {
    for (int column = 1; column <= n; column++) {
        if (place(row, column)) {
            board[row] = column;
            if (row == n)
                print(n);
            else
                queen(row + 1, n);
        }
    }
}

int main() {
    int n;
    printf("Enter number of Queens: ");
    scanf("%d", &n);
    queen(1, n);
    return 0;
}
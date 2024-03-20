/*Aim: To perform fractional knapsack

Complexity: O(nlog(m*n))
> here "m" is the length of the first string and "n" is the length of the second string.

Sample Input:

Enter first string: AGGTAB
Enter second string: GXTXAYB


PSEUDO CODE :

LCS(X, Y, m, n)
Step 1. Initialize a 2D array L of dimensions (m+1) x (n+1)
Step 2. For i = 0 to m:
1. Set L[i][0] = 0
Step 3. For j = 0 to n:
1. Set L[0][j] = 0
Step 4. For i = 1 to m:
1. For j = 1 to n:
1. If X[i-1] = Y[j-1]:
1. L[i][j] = L[i-1][j-1] + 1
2. Else:
1. L[i][j] = max(L[i-1][j], L[i][j-1])
Step 5. Return L[m][n] as the length of LCS

*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int max(int a, int b) {
    return (a > b) ? a : b;
}

void printLCS(char *X, char *Y, int m, int n) {
    int L[m + 1][n + 1];

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            if (i == 0 || j == 0)
                L[i][j] = 0;
            else if (X[i - 1] == Y[j - 1])
                L[i][j] = L[i - 1][j - 1] + 1;
            else
                L[i][j] = max(L[i - 1][j], L[i][j - 1]);
        }
    }

    int index = L[m][n];

    char lcs[index + 1];
    lcs[index] = '\0';

    int i = m, j = n;
    while (i > 0 && j > 0) {
        if (X[i - 1] == Y[j - 1]) {
            lcs[index - 1] = X[i - 1];
            i--;
            j--;
            index--;
        } else if (L[i - 1][j] > L[i][j - 1])
            i--;
        else
            j--;
    }

    printf("LCS of %s and %s is %s\n", X, Y, lcs);
}

int main() {
    char X[100], Y[100];
    printf("Enter first string: ");
    scanf("%s", X);
    printf("Enter second string: ");
    scanf("%s", Y);
    int m = strlen(X);
    int n = strlen(Y);
    printLCS(X, Y, m, n);
    return 0;
}
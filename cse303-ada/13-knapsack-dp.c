/*Aim: To perform 0/1 Knapsack using dynamic programming

Complexity: O(N*W)

Sample Input:

Enter the number of items: 3
Enter the values of the items:
60 100 120
Enter the weights of the items:
10 20 30
Enter the maximum weight capacity of the knapsack: 50



PSEUDO CODE :

KnapsackDP(values[], weights[], n, W)
Step 1. Create a 2D array dp of size (n+1) x (W+1) and initialize all values to 0.
Step 2. For i from 0 to n:
For w from 0 to W:
1. If i = 0 or w = 0, set dp[i][w] = 0.
2. Else if weights[i-1] <= w, set dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] +
values[i-1]).
3. Else, set dp[i][w] = dp[i-1][w].
Step 3. Return dp[n][W] as the maximum value that can be obtained.

*/

#include <stdio.h>
int max(int a, int b)
{
    return (a > b) ? a : b;
}
int knapsack(int n, int weights[], int values[], int W)
{
    int dp[n + 1][W + 1];
    for (int i = 0; i <= n; i++)
    {
        for (int w = 0; w <= W; w++)
        {
            if (i == 0 || w == 0)
                dp[i][w] = 0;
            else if (weights[i - 1] <= w)
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1]);
            else
                dp[i][w] = dp[i - 1][w];
        }
    }
    return dp[n][W];
}
int main()
{
    int n, W;
    printf("Enter the number of items: ");
    scanf("%d", &n);
    int values[n], weights[n];
    printf("Enter the values of the items:\n");
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &values[i]);
    }
    printf("Enter the weights of the items:\n");
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &weights[i]);
    }
    printf("Enter the maximum weight capacity of the knapsack: ");
    scanf("%d", &W);
    int result = knapsack(n, weights, values, W);
    printf("Maximum value in the knapsack: %d\n", result);
    return 0;
}
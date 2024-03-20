/*Aim: To perform fractional knapsack

Complexity: O(n)
> here "m" is the length of the first string and "n" is the length of the second string.

Sample Input:

Enter number of vertices: 5
Enter number of edges: 4
0 1
0 2
1 3
2 4



PSEUDO CODE :

DFS(graph, node):
Step 1. Create an empty stack and push the starting node onto the stack.
Step 2. Create an empty set to keep track of visited nodes.
Step 3. While the stack is not empty:
Pop a node from the stack and mark it as visited.
Process the node (e.g., print it or perform some operation).
Get the neighbors of the node.
For each neighbor:
If the neighbor has not been visited:
Push the neighbor onto the stack.
Step 4. Repeat until the stack is empty or the desired condition is met.

*/

#include <stdio.h>
#include <stdlib.h>

#define MAX 100

int adj[MAX][MAX]; // Adjacency matrix
int visited[MAX];  // Visited array

void DFS(int v, int n) {
    visited[v] = 1;
    printf("%d ", v);

    for (int i = 0; i < n; i++) {
        if (adj[v][i] == 1 && !visited[i]) {
            DFS(i, n);
        }
    }
}

int main() {
    int n, e, x, y;
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    printf("Enter number of edges: ");
    scanf("%d", &e);

    for (int i = 0; i < e; i++) {
        scanf("%d %d", &x, &y);
        adj[x][y] = 1;
    }

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            DFS(i, n);
        }
    }

    return 0;
}
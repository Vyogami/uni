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

BFS(graph, start):
Step 1. Create an empty queue and enqueue the starting node onto the queue.
Step 2. Create an empty set to keep track of visited nodes.
Step 3. While the queue is not empty:
Dequeue a node from the queue and mark it as visited.
Process the node (e.g., print it or perform some operation).
Get the neighbors of the node.
For each neighbor:
If the neighbor has not been visited:
Enqueue the neighbor onto the queue.
Step 4. Repeat until the queue is empty or the desired condition is met

*/


#include <stdio.h>
#include <stdlib.h>

#define MAX 100

int adj[MAX][MAX];
int visited[MAX];

void BFS(int start, int n) {
    int queue[MAX], rear = -1, front = -1, v;
    queue[++rear] = start;
    visited[start] = 1;

    while (rear != front) {
        v = queue[++front];
        printf("%d ", v);

        for (int i = 0; i < n; i++) {
            if (adj[v][i] && !visited[i]) {
                queue[++rear] = i;
                visited[i] = 1;
            }
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
            BFS(i, n);
        }
    }

    return 0;
}
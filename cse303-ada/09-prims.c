/*Aim: To perform prims algorithm

Complexity: O(m+nlog(n))

Sample Input:

Enter the number of vertices: 5
Enter the adjacency matrix for the graph:
0 2 0 6 0
2 0 3 8 5
0 3 0 0 7
6 8 0 0 9
0 5 7 9 0

Pseudo Code:

Prim'sAlgorithm(Graph G):
Step 1: Initialize an empty set 'MST' to store the minimum spanning tree
Step 2: Choose a starting vertex 'startVertex'
Step 3: Initialize a priority queue 'pq' to store edges with their weights
Step 4: Add all edges of 'startVertex' to 'pq' with their weights
Step 5: Mark 'startVertex' as visited
Step 6: While 'pq' is not empty:
    Step 7: Remove the edge with the smallest weight from 'pq', let it be 'currentEdge'
    Step 8: If 'currentEdge' connects a vertex not in MST to one in MST:
        Step 9: Add 'currentEdge' to 'MST'
        Step 10: Add all edges of 'currentEdge's destination vertex to 'pq' if not visited
        Step 11: Mark 'currentEdge's destination vertex as visited
Step 12: Return 'MST'


*/

// Write a program to implement prim's algorithm using greedy approach
#include <stdio.h>
#include <limits.h>

#define V 100

int minKey(int key[], int mstSet[], int vertices) {
    int min = INT_MAX, min_index;

    for (int v = 0; v < vertices; v++) {
        if (!mstSet[v] && key[v] < min) {
            min = key[v];
            min_index = v;
        }
    }

    return min_index;
}

void printMST(int parent[], int graph[V][V], int vertices) {
    printf("Edge \tWeight\n");
    for (int i = 1; i < vertices; i++) {
        printf("%d - %d \t%d\n", parent[i], i, graph[i][parent[i]]);
    }
}

void primMST(int graph[V][V], int vertices) {
    int parent[V];
    int key[V];
    int mstSet[V];

    for (int i = 0; i < vertices; i++) {
        key[i] = INT_MAX;
        mstSet[i] = 0;
    }

    key[0] = 0;
    parent[0] = -1;

    for (int count = 0; count < vertices - 1; count++) {
        int u = minKey(key, mstSet, vertices);
        mstSet[u] = 1;

        for (int v = 0; v < vertices; v++) {
            if (graph[u][v] && !mstSet[v] && graph[u][v] < key[v]) {
                parent[v] = u;
                key[v] = graph[u][v];
            }
        }
    }

    printMST(parent, graph, vertices);
}

int main() {
    int vertices;

    printf("Enter the number of vertices: ");
    scanf("%d", &vertices);

    int graph[V][V];

    printf("Enter the adjacency matrix for the graph:\n");
    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
            scanf("%d", &graph[i][j]);
        }
    }

    printf("Minimum Spanning Tree (MST) using Prim's Algorithm:\n");
    primMST(graph, vertices);

    return 0;
}

/*Aim: To perform dijkstras algorithm

Complexity: O(log(v^2))

Sample Input:

Enter number of vertices: 5
Enter graph data in matrix form:
0 3 0 4 0
3 0 1 2 0
0 1 0 0 7
4 2 0 0 5
0 0 7 5 0

Pseudo Code:

Dijkstra(Graph,source)

Step 1: Initialize distances[] and visited[] arrays
Step 2: Set distance from source to source as 0
Step 3: Set distance to all other vertices as infinity
Step 4: While there are unvisited vertices, do Steps 5-10
Step 5: Find the vertex 'u' in the unvisited set with the smallest distance
Step 6: Mark 'u' as visited
Step 7: For each neighbor 'v' of 'u', do Steps 8-10
Step 8: Calculate tentative distance from source to 'v'
Step 9: If tentative distance < distance[v], update distance[v] to tentative distance
Step 10: End loop over neighbors
Step 11: End loop over unvisited vertices
Step 12: Return distances[] as the shortest distances from the source vertex


*/

#include <stdio.h>
#include <limits.h>
#include <stdbool.h>

#define MAX_V 100

int minDistance(int dist[], bool sptSet[], int V) {
    int min = INT_MAX, min_index;
    for (int v = 0; v < V; v++) {
        if (sptSet[v] == false && dist[v] <= min) {
            min = dist[v];
            min_index = v;
        }
    }
    return min_index;
}

void printSolution(int dist[], int V) {
    printf("Vertex \t Distance from Source\n");
    for (int i = 0; i < V; i++) {
        printf("%d \t\t %d\n", i, dist[i]);
    }
}

void dijkstra(int graph[MAX_V][MAX_V], int src, int V) {
    int dist[MAX_V];
    bool sptSet[MAX_V];

    for (int i = 0; i < V; i++) {
        dist[i] = INT_MAX;
        sptSet[i] = false;
    }

    dist[src] = 0;

    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, sptSet, V);
        sptSet[u] = true;

        for (int v = 0; v < V; v++) {
            if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }

    printSolution(dist, V);
}

int main() {
    int V;
    printf("Enter number of vertices: ");
    scanf("%d", &V);

    int graph[MAX_V][MAX_V];

    printf("Enter graph data in matrix form:\n");
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            scanf("%d", &graph[i][j]);
        }
    }

    dijkstra(graph, 0, V);

    return 0;
}
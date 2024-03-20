/*Aim: To perform fractional knapsack

Complexity: O(nlog(n))

Sample Input:

Enter the number of items: 3
Enter the knapsack capacity: 50

Enter the weight and value of each item:
Item 1 - Weight: 10
Item 1 - Value: 60
Item 2 - Weight: 20
Item 2 - Value: 100
Item 3 - Weight: 30
Item 3 - Value: 120


PSEUDO CODE :

FractionalKnapsack(Item[] items, Value capacity)
Step 1: Sort the items based on their value-to-weight ratios in non-increasing order
Step 2: Initialize totalValue = 0.0
Step 3: For each item in items, do the following:
Step 4: If the item's weight is less than or equal to the remaining capacity:
Step 5: Take the entire item and add its value to totalValue
Step 6: Reduce the capacity by the item's weight
Step 7: Otherwise, if the item's weight is greater than the remaining capacity:
Step 8: Take a fraction of the item that fits into the remaining capacity
Step 9: Calculate the fraction as (remaining capacity / item's weight)
Step 10: Add the fraction * item's value to totalValue
Step 11: Break the loop
Step 12: Return totalValue

*/


#include <stdio.h>
#include <stdlib.h>

// Structure to represent an item
struct Item {
    int weight;
    int value;
};

// Function to compare items based on their value-to-weight ratio
int compare(const void *a, const void *b) {
    double ratioA = ((double)((struct Item *)a)->value) / ((struct Item *)a)->weight;
    double ratioB = ((double)((struct Item *)b)->value) / ((struct Item *)b)->weight;

    if (ratioA < ratioB)
        return 1;
    else if (ratioA > ratioB)
        return -1;
    else
        return 0;
}

// Function to perform Fractional Knapsack
double fractionalKnapsack(struct Item items[], int n, int capacity) {
    // Sort items based on their value-to-weight ratio
    qsort(items, n, sizeof(struct Item), compare);

    double totalValue = 0.0; // Total value in the knapsack

    for (int i = 0; i < n; i++) {
        if (capacity <= 0)
            break;

        // Take the whole item if it fits, otherwise take a fraction of it
        double fraction = (capacity < items[i].weight) ? ((double)capacity / items[i].weight) : 1.0;

        // Update the capacity and total value
        capacity -= fraction * items[i].weight;
        totalValue += fraction * items[i].value;
    }

    return totalValue;
}

int main() {
    int n, capacity;

    // Input: Number of items and knapsack capacity
    printf("Enter the number of items: ");
    scanf("%d", &n);
    printf("Enter the knapsack capacity: ");
    scanf("%d", &capacity);

    // Input: Weight and value of each item
    struct Item items[n];
    printf("\nEnter the weight and value of each item:\n");
    for (int i = 0; i < n; i++) {
        printf("Item %d - Weight: ", i + 1);
        scanf("%d", &items[i].weight);
        printf("Item %d - Value: ", i + 1);
        scanf("%d", &items[i].value);
        printf("\n");
    }

    // Perform Fractional Knapsack
    double maxValue = fractionalKnapsack(items, n, capacity);

    // Output: Maximum value that can be obtained
    printf("\nMaximum value in the knapsack: %.2f\n", maxValue);

    return 0;
}

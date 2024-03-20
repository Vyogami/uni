/*Aim: To perform binary search

Complexity: O(log(n))

Sample Input:

Enter the size of the sorted array: 7
Enter the sorted elements of the array:
array[0]: 10
array[1]: 20
array[2]: 30
array[3]: 40
array[4]: 50
array[5]: 60
array[6]: 70
Enter the element to be searched: 40


Enter the element to be searched: 35

Pseudo Code:

Binary Search (Array A, Value x, low, high)
Step 1: If low is greater than high, go to Step 10
Step 2: Set mid to (low + high) / 2
Step 3: If A[mid] equals x, go to Step 8
Step 4: If A[mid] is less than x, recursively call Binary Search with (A, x, mid + 1, high)
Step 5: If A[mid] is greater than x, recursively call Binary Search with (A, x, low, mid - 1)
Step 6: Exit
Step 7: Print Element x Found at index mid and go to Step 9
Step 8: Exit
Step 9: Exit
Step 10: Print Element not found and go to Step 9

*/

#include <stdio.h>

// Function to perform Binary Search recursively
int binarySearchRecursive(int arr[], int low, int high, int key) {
    if (low <= high) {
        int mid = low + (high - low) / 2;

        if (arr[mid] == key) {
            return mid; // Element found, return its index
        } else if (arr[mid] < key) {
            return binarySearchRecursive(arr, mid + 1, high, key); // Search in the right half
        } else {
            return binarySearchRecursive(arr, low, mid - 1, key); // Search in the left half
        }
    }

    return -1; // Element not found
}

int main() {
    int size, key;

    // Input: Size of the array
    printf("Enter the size of the sorted array: ");
    scanf("%d", &size);

    // Input: Elements of the sorted array
    int arr[size];
    printf("\nEnter the sorted elements of the array:\n");
    for (int i = 0; i < size; i++) {
        printf("array[%d]: ", i);
        scanf("%d", &arr[i]);
    }

    // Input: Element to be searched
    printf("\nEnter the element to be searched: ");
    scanf("%d", &key);

    // Perform Binary Search recursively
    int result = binarySearchRecursive(arr, 0, size - 1, key);

    // Output
    if (result != -1) {
        printf("Element %d found at index %d.\n", key, result);
    } else {
        printf("Element %d not found in the array.\n", key);
    }

    return 0;
}

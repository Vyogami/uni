/*Aim: To perform selection sort

Complexity: O(log(n^2))

Sample Input:

Enter the size of the array: 6

Enter the elements of the array:
array[0]: 30
array[1]: 10
array[2]: 50
array[3]: 20
array[4]: 40
array[5]: 60

PSEUDO CODE :

SelectionSort(Array A, Value length)
Step 1: For i = 0 to length - 1
Step 2: Set min_index = i
Step 3: For j = i + 1 to length
Step 4: If A[j] < A[min_index], set min_index = j
Step 5: Swap A[i] and A[min_index]
Step 6: Array A is now sorted in ascending order

*/

#include <stdio.h>

// Function to perform Selection Sort
void selectionSort(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        // Find the minimum element in the unsorted part of the array
        int minIndex = i;
        for (int j = i + 1; j < size; j++) {
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }

        // Swap the found minimum element with the first element
        int temp = arr[minIndex];
        arr[minIndex] = arr[i];
        arr[i] = temp;
    }
}

int main() {
    int size;

    // Input: Size of the array
    printf("Enter the size of the array: ");
    scanf("%d", &size);

    // Input: Elements of the array
    int arr[size];
    printf("\nEnter the elements of the array:\n");
    for (int i = 0; i < size; i++) {
        printf("array[%d]: ", i);
        scanf("%d", &arr[i]);
    }

    // Perform Selection Sort
    selectionSort(arr, size);

    // Output: Sorted array
    printf("\nSorted array after Selection Sort:\n");
    for (int i = 0; i < size; i++) {
        printf("array[%d]: %d\n", i, arr[i]);
    }

    return 0;
}

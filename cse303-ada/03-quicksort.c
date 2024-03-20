/*Aim: To perform quicksort

Complexity:
    best: O(nlog(n))
    worst:O(n^2)

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

Quicksort (Array A, Value low, Value high)
Step 1: If low is greater than or equal to high, return (base case)
Step 2: Choose a pivot element from A (e.g., A[high])
Step 3: Partition the array A into two sub-arrays:
- Elements less than the pivot go to the left sub-array
- Elements greater than the pivot go to the right sub-array
Step 4: Recursively call QuickSort on the left sub-array: QuickSort(A, low, pivot_index - 1)
Step 5: Recursively call QuickSort on the right sub-array: QuickSort(A, pivot_index + 1,
high)
Step 6: Array A is now sorted in ascending order between low and high indices (no action
needed)
Step 7: Return
*/

#include <stdio.h>

// Function to partition the array and return the pivot index
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            // Swap arr[i] and arr[j]
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    // Swap arr[i+1] and arr[high] (put the pivot in its correct position)
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;

    return i + 1;
}

// Function to implement QuickSort
void quickSort(int arr[], int low, int high) {
    if (low < high) {
        // Partition the array, arr[p] is now at the correct position
        int pivotIndex = partition(arr, low, high);

        // Recursively sort the elements before and after the pivot
        quickSort(arr, low, pivotIndex - 1);
        quickSort(arr, pivotIndex + 1, high);
    }
}

int main() {
    int size;

    printf("Enter the size of the array: ");
    scanf("%d", &size);

    int arr[size];
    printf("\nEnter the elements of the array:\n");
    for (int i = 0; i < size; i++) {
        printf("array[%d]: ", i);
        scanf("%d", &arr[i]);
    }

    quickSort(arr, 0, size - 1);

    printf("\nSorted array after QuickSort:\n");
    for (int i = 0; i < size; i++) {
        printf("array[%d]: %d\n", i, arr[i]);
    }

    return 0;
}

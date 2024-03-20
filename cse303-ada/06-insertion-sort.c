/*Aim: To perform insertion sort

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

InsertionSort(Array A, Value length)
Step 1: For i = 1 to length - 1
Step 2: Set current_element = A[i]
Step 3: Set j = i - 1
Step 4: While j >= 0 and A[j] > current_element
Step 5: Shift A[j] to A[j + 1]
Step 6: Decrement j
Step 7: Place current_element in A[j + 1]
Step 8: Array A is now sorted in ascending order

*/


#include <stdio.h>

// Function to perform Insertion Sort
void insertionSort(int arr[], int size) {
    for (int i = 1; i < size; i++) {
        int key = arr[i];
        int j = i - 1;

        // Move elements of arr[0..i-1] that are greater than key to one position ahead of their current position
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
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

    // Perform Insertion Sort
    insertionSort(arr, size);

    // Output: Sorted array
    printf("\nSorted array after Insertion Sort:\n");
    for (int i = 0; i < size; i++) {
        printf("array[%d]: %d\n", i, arr[i]);
    }

    return 0;
}

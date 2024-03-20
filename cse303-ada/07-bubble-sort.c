/*Aim: To perform bubble sort

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

BubbleSort(Array A, Value length)
Step 1: For i = 0 to length - 1
Step 2: For j = 0 to length - i - 1
Step 3: If A[j] > A[j + 1], swap A[j] and A[j + 1]
Step 4: Array A is now sorted in ascending order

*/

#include <stdio.h>

// Function to perform Bubble Sort
void bubbleSort(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            // Swap if the element found is greater than the next element
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
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

    // Perform Bubble Sort
    bubbleSort(arr, size);

    // Output: Sorted array
    printf("\nSorted array after Bubble Sort:\n");
    for (int i = 0; i < size; i++) {
        printf("array[%d]: %d\n", i, arr[i]);
    }

    return 0;
}

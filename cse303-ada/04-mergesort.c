/*Aim: To perform mergesort

Complexity: O(nlog(n))

Sample Input:

Enter the size of the array: 6

Enter the elements of the array:
array[0]: 30
array[1]: 10
array[2]: 50
array[3]: 20
array[4]: 40
array[5]: 60

Mergesort(Array A, Value low, Value high)
Step 1: If low is greater than or equal to high, return (base case)
Step 2: Set mid to (low + high) / 2
Step 3: Recursively call MergeSort on the left half: MergeSort(A, low, mid)
Step 4: Recursively call MergeSort on the right half: MergeSort(A, mid + 1, high)
Step 5: Merge the two sorted halves:
- Create temporary arrays for the left and right halves
- Copy the elements from A[low:mid] to the left array
- Copy the elements from A[mid+1:high] to the right array
- Merge the left and right arrays back into A[low:high] in sorted order
Step 6: Return

*/

#include <stdio.h>

// Function to merge two subarrays of arr[]
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Create temporary arrays
    int leftArray[n1], rightArray[n2];

    // Copy data to temporary arrays leftArray[] and rightArray[]
    for (int i = 0; i < n1; i++) {
        leftArray[i] = arr[left + i];
    }
    for (int j = 0; j < n2; j++) {
        rightArray[j] = arr[mid + 1 + j];
    }

    // Merge the temporary arrays back into arr[left..right]
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (leftArray[i] <= rightArray[j]) {
            arr[k] = leftArray[i];
            i++;
        } else {
            arr[k] = rightArray[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of leftArray[], if there are any
    while (i < n1) {
        arr[k] = leftArray[i];
        i++;
        k++;
    }

    // Copy the remaining elements of rightArray[], if there are any
    while (j < n2) {
        arr[k] = rightArray[j];
        j++;
        k++;
    }
}

// Function to implement MergeSort
void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        // Same as (left + right) / 2, but avoids overflow for large left and right
        int mid = left + (right - left) / 2;

        // Sort first and second halves
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        // Merge the sorted halves
        merge(arr, left, mid, right);
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

    // Perform MergeSort
    mergeSort(arr, 0, size - 1);

    // Output: Sorted array
    printf("\nSorted array after MergeSort:\n");
    for (int i = 0; i < size; i++) {
        printf("array[%d]: %d\n", i, arr[i]);
    }

    return 0;
}

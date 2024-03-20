/*Aim: To perform binary search iterative

Complexity: O(log(n))

Sample Input:

Enter the size of the sorted array: 8

Enter the sorted elements of the array:
array[0]: 5
array[1]: 15
array[2]: 25
array[3]: 35
array[4]: 45
array[5]: 55
array[6]: 65
array[7]: 75

Enter the element to be searched: 35

Pseudo Code:

Binary Search (Array A, Value x)
Step 1: Set low to 1
Step 2: Set high to n (where n is the number of elements in the array)
Step 3: While low is less than or equal to high, do Steps 4 to 6
Step 4: Set mid to (low + high) / 2
Step 5: If A[mid] equals x, go to Step 7
Step 6: If A[mid] is less than x, set low to mid + 1 and go to Step 3
Step 7: If A[mid] is greater than x, set high to mid - 1 and go to Step 3
Step 8: Print Element x Found at index mid and go to Step 9
Step 9: Exit
Step 10: Print Element not found and go to Step 9

*/


#include <stdio.h>

int binarySearch(int arr[], int size, int key) {
    int low = 0, high = size - 1;

    while (low <= high) {
        int mid = low + (high - low) / 2;

        if (arr[mid] == key) {
            return mid;
        } else if (arr[mid] < key) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return -1;
}

int main() {
    int size, key;

    printf("Enter the size of the sorted array: ");
    scanf("%d", &size);

    int arr[size];
    printf("\nEnter the sorted elements of the array:\n");
    for (int i = 0; i < size; i++) {
        printf("array[%d]: ", i);
        scanf("%d", &arr[i]);
    }

    printf("\nEnter the element to be searched: ");
    scanf("%d", &key);

    int result = binarySearch(arr, size, key);

    if (result != -1) {
        printf("Element %d found at index %d.\n", key, result);
    } else {
        printf("Element %d not found in the array.\n", key);
    }

    return 0;
}

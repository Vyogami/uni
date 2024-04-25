# A sequence of integers of even length is said to be left-heavy if the sum of the terms in the left-half of the sequence
# is greater than the sum of the terms in the right half. It is termed right-heavy if the sum of the second half is greater
# than the first half. It is said to be balanced if both the sums are equal.

# Accept a sequence of comma-separated integers as input. Determine if the sequence is left-heavy, right-heavy
# or balanced and print this as the output.

# Taking user input
array = [int(i) for i in input("Enter the numbers: ").strip().split(',')]

# Calculating the starting index of right side and the summation of both the sides
mid = len(array) // 2
left = sum(array[:mid])
right = sum(array[mid:])

# Determining which side is heavier
if(left > right):
    print("left-heavy")
elif(right > left):
    print("right-heavy")
else:
    print("balanced")


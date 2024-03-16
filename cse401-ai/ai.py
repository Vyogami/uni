# Standard output
print("\n\nStandard output:")
print("Hello, world!")

# f strings
print("\n\nf strings:")
name = "Alice"
age = 30
print(f"My name is {name} and I am {age} years old.")

# b strings
print("\n\nb strings:")
b_string = b"Hello"
print(b_string)

# Output formatting
print("\n\nOutput formatting:")
num1 = 10
num2 = 20
print("The sum of {} and {} is {}".format(num1, num2, num1 + num2))

# Standard input
print("\n\nStandard input:")
user_input = input("Enter your name: ")
print("Hello,", user_input)

# Variables
print("\n\nVariables:")
x = 10
y = 20
result = x + y
print(result)

# Naming conventions and constants
print("\n\nNaming conventions and constants:")
MAX_VALUE = 100
print(MAX_VALUE)

# Scope
print("\n\nScope:")
def func():
    local_var = 10
    print(local_var)

func() 

# Data types
print("\n\nData types:")
# (Examples for int, float, string, bool, complex numbers, list, tuples, dictionaries, sets, byte, bytearray)

# int
print("\n\nint:")
my_int = 10
print(my_int)

# float
print("\n\nfloat:")
my_float = 3.14
print(my_float)

# string
print("\n\nstring:")
my_string = "Hello"
print(my_string)

# bool
print("\n\nbool:")
my_bool = True
print(my_bool)

# complex numbers
print("\n\ncomplex numbers:")
my_complex = 3 + 4j
print(my_complex)

# list
print("\n\nlist:")
my_list = [1, 2, 3]
print(my_list)

# tuples
print("\n\ntuples:")
my_tuple = (1, 2, 3)
print(my_tuple)

# dictionaries
print("\n\ndictionaries:")
my_dict = {"a": 1, "b": 2}
print(my_dict)

# sets
print("\n\nsets:")
my_set = {1, 2, 3}
print(my_set)

# byte
print("\n\nbyte:")
my_byte = b'Hello'
print(my_byte)

# bytearray
print("\n\nbytearray:")
my_bytearray = bytearray(b'Hello')
print(my_bytearray)

# array
print("\n\narray:")
from array import array
my_array = array('i', [1, 2, 3])
print(my_array)

# imports
print("\n\nimports:")
import math
print(math.pi)

# numpy array
print("\n\nnumpy array:")
import numpy as np
my_np_array = np.array([1, 2, 3])
print(my_np_array)

# basic list methods
print("\n\nbasic list methods:")
my_list = [1, 2, 3]
my_list.append(4)
print(my_list)

# basic array methods
print("\n\nbasic array methods:")
my_array.append(4)
print(my_array)

# basic tuple methods
print("\n\nbasic tuple methods:")
my_tuple = (1, 2, 3)
print(my_tuple.index(2))

# basic dictionaries methods
print("\n\nbasic dictionaries methods:")
my_dict = {"a": 1, "b": 2}
print(my_dict.keys())

# basic sets methods
print("\n\nbasic sets methods:")
my_set.add(4)
print(my_set)

# slicing
print("\n\nslicing:")
# (Examples for string, tuples, array)
# string
my_string = "Hello, world!"
print(my_string[2:6])

# tuples
print("\n\ntuples:")
my_tuple = (1, 2, 3, 4, 5)
print(my_tuple[2:4])

# array
print("\n\narray:")
print(my_array[1:3])

# string manipulations (basic string methods)
print("\n\nstring manipulations (basic string methods):")
my_string = "Hello, world!"
print(my_string.upper())

# generators
print("\n\ngenerators:")
def my_generator():
    yield 1
    yield 2
    yield 3

gen = my_generator()
print(next(gen))
print(next(gen))
print(next(gen))

# Control flow
print("\n\nControl flow:")
# if-else
x = 10
if x > 5:
    print("x is greater than 5")
else:
    print("x is less than or equal to 5")

# Loops
print("\n\nLoops:")
# for loop
for i in range(5):
    print(i)

# while loop
print("\n\nwhile loop:")
count = 0
while count < 5:
    print(count)
    count += 1

# functions
print("\n\nfunctions:")
def add(a, b):
    return a + b

print(add(2, 3))

# classes
print("\n\nclasses:")
class MyClass:
    def __init__(self, x):
        self.x = x
    
    def display(self):
        print(self.x)

obj = MyClass(10)
obj.display()

# decorators
print("\n\ndecorators:")
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

# context manager
print("\n\ncontext manager:")
with open('file.txt', 'w') as f:
    f.write('Hello, world!')

# error handling
print("\n\nerror handling:")
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero.")

# para is a sequence of space-separated words. All ords will be in lower case. There will be a single space between
# consecutive words. The string has no other special characters other than the space.

# Write a function named exact_count that accepts the string para and a p0sitive integer n as arguments. You have to
# return True if there is at least one ord in para that occurs exactly n times, and False otherwise.
# You do not have to accept input from the user or print output to the console. You just have to write the function
# definition.

def exac_count(para: str, n: int):
    words = para.split(' ')
    for word in words:
        if(para.count(word) == n):
            return True
    
    return False



if __name__ == "__main__":
    para = input("Enter the paragraph: ")
    n = int(input("Enter the number of occurance: "))
    print(exac_count(para, n))

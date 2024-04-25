# A data entry operator has a fauty keypoard. Ihe keys U and I are very unreliable. sometimes they work, sometimes they
# don't. While entering phone numbers into a database, the operator uses the letter | as a replacement for l and 'o' as a
# replacement for 0 whenever these binary digits let him down. Both T and 'o are in lower case. T is the first letter of the
# Word land, and not capital T.
#
# Accept a ten-digit number as input. Find the number of places where the numbers 0 and 1 have been replaced by letters.
# If there are no such replacements, print the string No mistakes. If not, print the number of mistakes (replacements)
# and in the next line, print the correct phone number

string = input("Enter the number: ")
mistakes = string.count('o') + string.count('l')
print(f"{mistakes} mistakes\n{string}" if mistakes > 0 else "NO Mistakes")
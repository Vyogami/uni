from datetime import datetime
date = datetime.strptime(input(), '%d-%m-%Y')
print(date.year, end="")

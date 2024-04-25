from datetime import datetime
print(datetime.strptime(input(), '%d-%m-%Y').year, end="")

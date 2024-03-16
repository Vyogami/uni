# Gurunath is a popular store inside IlT Madras. Among other things, it sells stationary items. The owner of the stores
# gives you the list of transactions that have happened in a day. Each transcation comes with a unique transaction ID. He
# wants to estaimte the cost of each transaction. Can you help him out?
# The list tIans is a list of transactions that happened at the shop in a given day. Each element of this list is a dictionary.
#
# The details of one such transaction is given below:
# {
# "TID": 'Gurunath_8528',
# "Items": I
    # {'Name': "Notebook", 'Price': 50, Qty': 4,
    # {'Name': 'Pencil', 'Price': 10, 'Qty': 1},
    # {'Name': 'Eraser', Price': 15, Qty': 1},
    # {'Name': "Flle', Price': 80, "Qty': 1}
# }
# 
# Does this remind you of the shopping bills dataset from CT?
# 
# Write a function named get_summary that accepts the list trans as argument. It should return a list of dictionaries.
# Each dictionary should have two keys: "TID" and "Cost". For example, one of the elements of the list would be ass
# follows:
#
# {
#     "Cost": 305,
#     "TID" "Gurunath_8528
# }
# The computation of the cost is given below:
# 50 x 4+10 x 1+15 x 1+80 x1= 305
# (1) The order of elements in the returned list should be the same as the order in the input list. That is, the i element in
# the returned list should correspond to the transaction cost of the ih element in trans
# (2) You do not have to accept input from the user or print the output to the console. You just have to write the function
# definition.

def get_summary(transaction: dict):
    items = transaction["Items"]
    result = {
        "Cost": 0,
        "TID": transaction["TID"]
    }
    
    for item in items:
        result["Cost"] += item["Price"] * item["Qty"]
    return result
    


if __name__ == "__main__":
    transaction = {
        'TID': 'Gurunath_8528', 
        'Items': [
            {'Name': 'Notebook', 'Price': 50, 'Qty': 4}, 
            {'Name': 'Pencil', 'Price': 10, 'Qty': 1}, 
            {'Name': 'Eraser', 'Price': 15, 'Qty': 1}, 
            {'Name': 'File', 'Price': 80, 'Qty': 1}
        ]
    }
    
    print(get_summary(transaction))
    

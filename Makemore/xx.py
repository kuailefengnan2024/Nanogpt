import sys

my_list = [1, 2, 3]
my_tuple = (1, 2, 3)

print(f"列表大小: {sys.getsizeof(my_list)}")
print(f"元组大小: {sys.getsizeof(my_tuple)}")

my_list_empty = []
my_tuple_empty = ()

print(f"空列表大小: {sys.getsizeof(my_list_empty)}")
print(f"空元组大小: {sys.getsizeof(my_tuple_empty)}")
variable_table=[
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
]

your_function = [0, 1, 4, 5, 7, 9, 11, 13]
logic_function = []

# Formation of a logical function in a form understandable to a computer
for function_index in your_function:
    variable_group = []
    for table_row in range(len(variable_table)):
        if function_index == table_row:
            for boolean_index in range(len(variable_table[table_row])):
                variable_group.append((((boolean_index + 1) * 10) + variable_table[table_row][boolean_index]))
    logic_function.append(variable_group)

"""
We create arrays in which we will store the values of the logical function and the applicants,
from which we will form the final table for finding the coverage.
"""
initial_logic_function = logic_function[:]
initial_implicants = []
# You need to know the initial value of the group length, it = constant
group_lenght = len(logic_function[0])

# Repeat the loop until the group size is 1, which means that we have found all the implicants.
while group_lenght != 1:
    group_lenght -= 1
    """
    Formation of an initial terms table - a table with:
    1) the correct names of columns and rows,
    2) ones in the main diagonal and zeros in the rest of the cells.
    """
    initial_terms_table = [[0, logic_function[:]], ]
    for table_row in range(len(logic_function)):
        current_row = [logic_function[table_row][:]]
        for element in range(len(logic_function)):
            if element == table_row:
                current_row.append(1)
            else:
                current_row.append(0)
        initial_terms_table.append(current_row)

    # We finish the formation of this table, at the output we get a table with implicants in its required cells.
    for row in range(len(initial_terms_table)):
        if row == 0:
            continue
        for table_cell in range(len(initial_terms_table[row])):
            not_same_array = []
            if table_cell == 0 or initial_terms_table[row][table_cell] == 1:
                continue
            for element in initial_terms_table[row][0]:
                for secondary_element in initial_terms_table[0][1][table_cell - 1]:
                    if element != secondary_element and int(str(element)[0]) == int(str(secondary_element)[0]):
                        not_same_array.append(secondary_element)
            if len(not_same_array) == 1:
                initial_terms_table[row][table_cell] = initial_terms_table[0][1][table_cell - 1][:]
                initial_terms_table[row][table_cell].remove(not_same_array[0])
                for element in initial_terms_table[row][table_cell]:
                    if element not in initial_terms_table[row][0]:
                        initial_terms_table[row][table_cell] = 0

    """
    Now our logic_function is equal to the set of implicants.
    In the second big cycle, all repetitions of the implicants are removed.
    """
    logic_function = []
    for row in range(len(initial_terms_table)):
        if row == 0:
            continue
        for table_cell in range(len(initial_terms_table[row])):
            if initial_terms_table[row][table_cell] != 1 and initial_terms_table[row][table_cell] != 0:
                if len(initial_terms_table[row][table_cell]) == group_lenght:
                    logic_function.append(initial_terms_table[row][table_cell])
    changed = 1
    while changed == 1:
        changed = 0
        remove_list = []
        for group in range(len(logic_function)):
            for secondary_group in range(len(logic_function)):
                if group != secondary_group and logic_function[group] == logic_function[secondary_group] and logic_function[group] not in remove_list:
                    remove_list.append(logic_function[group])
                    changed = 1
        for group in remove_list:
            logic_function.remove(group)

    # At the end of the while loop, do not forget to add the resulting implicants to the array.
    for group in logic_function:
        initial_implicants.append(group)

"""
We form the final table for finding the coverage.
First, we form the names of columns and rows, and fill all other cells with zeros.
"""
initial_implicants_table = [[0, initial_logic_function[:]], ]
for table_row in range(len(initial_implicants)):
    current_row = [initial_implicants[table_row][:],]
    for element in range(len(initial_logic_function)):
        current_row.append(0)
    initial_implicants_table.append(current_row)

# Then we find occurrences and set ones on the corresponding cells.
for row in range(len(initial_implicants_table)):
    if row == 0:
        continue
    for table_cell in range(len(initial_implicants_table[row])):
        break_out_flag = False
        if table_cell == 0:
            continue
        for element in initial_implicants_table[row][0]:
            if element not in initial_implicants_table[0][1][table_cell - 1]:
                break_out_flag = True
                break
        if break_out_flag == True:
            continue
        initial_implicants_table[row][table_cell] = 1

# We translate the table into a matrix, make it suitable for the program for finding the coverage.
matrix = initial_implicants_table[:]
del matrix[0]
for row in matrix:
    print(row)

# We make coverage and row arrays of the appropriate size for the program to work.
Cover = []
Empty_row = []
for row in range(len(matrix[0]) - 1):
    Cover.append(1)
    Empty_row.append(0)

# The program for finding coverage has been slightly reworked.
current_amount_row = []
best_amount_row = []
best_amount = 1000
repeat_counter = 0
stop_recursion = False
def find_cover(last_row_counter, last_row_Temp):
    global best_amount, best_amount_row, repeat_counter, stop_recursion
    for row in range(last_row_counter, len(matrix)):
        repeat_counter += 1
        Temp = last_row_Temp[:]
        row_counter = last_row_counter + 1
        current_amount_row.append(row)
        for element in range(1, len(matrix[0])):
            if Temp[element - 1] == 0 and matrix[row][element] == 1:
                Temp[element - 1] = 1
        if Temp == Cover:
            if row_counter < best_amount:
                best_amount = row_counter
                best_amount_row = current_amount_row[:]
                if repeat_counter > (100 * len(your_function)):
                    stop_recursion = True
            current_amount_row.remove(row)
            continue
        else:
            if stop_recursion == False:
                find_cover(row_counter, Temp)
                current_amount_row.remove(row)
find_cover(0, Empty_row)

print("Знайдено найкраще рішення скорочення функції методом Квайна:")
print(f"{matrix[best_amount_row[0]][0]} ", end='')
for row in best_amount_row:
    if row == best_amount_row[0]:
        continue
    elif row != best_amount_row[-1]:
        print(f"V {matrix[row][0]} ", end='')
    else:
        print(f"V {matrix[row][0]}.")
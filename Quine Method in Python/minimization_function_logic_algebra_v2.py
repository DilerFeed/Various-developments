"""
This is a program for automatic minimization of logic algebra functions according to Quine's method.
The program is almost completely automatic - you just need to enter the numbers of sets of logical variables in the array below.
As a result of execution, you will receive arrays of two-digit numbers, which are actually variables from the task.
You can see the logic of converting these two-digit numbers into variables here:
10 - NOT x1; 11 - x1;
20 - NOT x2; 21 - x2;
30 - NOT x3; 31 - x3;
40 - NOT x4; 41 - x4.
Made for laboratory work 4 for discrete mathematics by a student of group 6KN-22b Ishchenko Gleb, VNTU, Ukraine.
"""

# Boolean variable set table
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

# Enter your value here!!!!!!!!!!!!!!!!
your_function = [0, 1, 4, 5, 7, 9, 11, 13]

logic_function = [] # Here we will write down all implicants

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

# -------------------------------- From this moment, the search for coverage by the method of core rows begins.

# We translate the table into a matrix, make it suitable for the program for finding the coverage.
matrix = []
for row in initial_implicants_table:
    matrix.append(row[:])
del matrix[0]
for row in range(len(matrix)):
    row_cost = len(matrix[row][0]) - 1
    matrix[row].append(row_cost)
    matrix[row].append((row + 1))
    del matrix[row][0]

# Additional matrix. Please note that we create all matrices using the slice method.
matrix_copy = []
for row in matrix:
    matrix_copy.append(row[:])
m_c_rows = len(matrix_copy)
m_c_columns = (len(matrix[0]) - 2)
Core_rows_array = []

def index_correction(index_array):
    for k in range(len(index_array)):
        if k != 0:
            index_array[k] -= k

def count_ones(row):
    counter = 0
    for element in range(len(row) - 2):
        if element == 1:
            counter += 1
    return counter

changed = True
# The method has been slightly improved, since we will often get matrices smaller than the minimum size (2 rows).
while changed == True and m_c_rows > 2:
    changed = False
    core_columns = []
    core_rows = []

    for column_index in range(m_c_columns):
        ones_counter = 0
        for row_index in range(m_c_rows):
            if matrix_copy[row_index][column_index] == 1:
                ones_counter += 1
                one_index = row_index
        if ones_counter == 1:
            core_columns.append(column_index)
            core_rows.append(one_index)
            changed = True

    for core_row in core_rows:
        Core_rows_array.append(matrix[core_row])

    core_rows.sort()
    index_correction(core_columns)
    index_correction(core_rows)

    for column_index in core_columns:
        for row_index in range(m_c_rows):
            del matrix_copy[row_index][column_index]
    for row_index in core_rows:
        del matrix_copy[row_index]
    m_c_rows = len(matrix_copy)
    m_c_columns = (len(matrix_copy[0]) - 2)

    anticore_rows = []
    for row_index in range(m_c_rows):
        if count_ones(matrix_copy[row_index]) == 0:
            anticore_rows.append(row_index)
            changed = True
    anticore_rows.sort()
    index_correction(anticore_rows)
    for row_index in anticore_rows:
        del matrix_copy[row_index]
    m_c_rows = len(matrix_copy)

    """
    Unlike Quine's original method, I use the full core rows method.
    This means that we are also looking for rows that are absorbed.
    This is a complex algorithm, but it will allow us to find the best possible solution as quickly as possible.
    The main advantage - most often the enumeration will not be needed.
    """
    absorbed_rows = []
    for main_row_index in range(m_c_rows):
        for secondary_row_index in range(m_c_rows):
            break_out_flag = False
            if main_row_index != secondary_row_index:
                for column_index in range(m_c_columns):
                    if matrix_copy[main_row_index][column_index] == 1 and matrix_copy[main_row_index][column_index] != matrix_copy[secondary_row_index][column_index]:
                        break_out_flag = True
                        break
                if break_out_flag:
                    continue
                if count_ones(matrix_copy[main_row_index]) <= count_ones(matrix_copy[secondary_row_index]) and main_row_index not in absorbed_rows and matrix_copy[main_row_index][-2] >= matrix_copy[secondary_row_index][-2]:
                    absorbed_rows.append(main_row_index)
                    changed = True

    index_correction(absorbed_rows)
    for row_index in absorbed_rows:
        del matrix_copy[row_index]
    m_c_rows = len(matrix_copy)

# This algorithm is almost perfect, but if necessary, this place can be improved if the problems are large.
Cover = [1, 1, 1, 1, 1, 1 ,1 , 1, 1]
Empty_row = [0, 0, 0, 0, 0, 0, 0 ,0, 0]
for element in range(9 - m_c_columns):
    del Cover[0]
    del Empty_row[0]

# We use the boundary enumeration method, if necessary. Most of the time it will just skip.
current_amount_row = []
best_amount_row = []
best_amount = 1000
current_cost_row = []
best_cost_row = []
best_cost = 1000
def find_cover(last_cost_counter, last_row_counter, last_row_Temp):
    global best_amount, best_amount_row, best_cost, best_cost_row
    for row in range(last_row_counter, m_c_rows):
        Temp = last_row_Temp[:]
        row_counter = last_row_counter + 1
        current_amount_row.append(row)
        current_cost_row.append(row)
        cost = matrix_copy[row][-2] + last_cost_counter
        for element in range(m_c_columns):
            if Temp[element] == 0 and matrix_copy[row][element] == 1:
                Temp[element] = 1
        if Temp == Cover:
            if row_counter < best_amount:
                best_amount = row_counter
                best_amount_row = current_amount_row[:]
            if cost < best_cost:
                best_cost = cost
                best_cost_row = current_cost_row[:]
            current_amount_row.remove(row)
            current_cost_row.remove(row)
            continue
        else:
            find_cover(cost, row_counter, Temp)
            current_amount_row.remove(row)
            current_cost_row.remove(row)


find_cover(0, 0, Empty_row)

for row in range(len(best_amount_row)):
    best_amount_row[row] = matrix_copy[best_amount_row[row]][-1]
for row in range(len(best_cost_row)):
    best_cost_row[row] = matrix_copy[best_cost_row[row]][-1]

for core_row in Core_rows_array:
    best_amount += 1
    best_amount_row.append(core_row[-1])
    best_cost += core_row[-2]
    best_cost_row.append(core_row[-1])
best_amount_row.sort()
best_cost_row.sort()

# -------------------------------- At this point, the best coverages have already been found for sure.

# As a result of my program, we will receive a lot of useful data. In accordance with Quine's method, we are only interested in the shortest one, we only display it.
print("The best solution for reducing the function using Quine's method was found:")
print(f"{initial_implicants_table[best_amount_row[0]][0]} ", end='')
for row in best_amount_row:
    if row == best_amount_row[0]:
        continue
    elif row != best_amount_row[-1]:
        print(f"V {initial_implicants_table[row][0]} ", end='')
    else:
        print(f"V {initial_implicants_table[row][0]}.")

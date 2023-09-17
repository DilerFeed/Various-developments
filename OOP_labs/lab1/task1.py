import datetime

class Students():
    def __init__(self):
        self.students = []
        
    def add_student(self, name):
        if len(name) > 0:
            self.students.append(name)
            
            print(f'New student "{name}" added with id {(len(self.students) - 1)}.')
        else:
            print("Error: Student name must exist")
            
    def print_student(self, id, name):
        found = False
        if id != '' and self.students[id] != 0:
            print(f'Found a student with name "{self.students[id]}" under id {id}!')
            found = True
        if name != '' and self.students.count(name) > 0:
            print(f'Found a student with name "{name}" under id {self.students.index(name)}!')
            found = True
        if found == False:
            print('No student with this name/id found')
            
    def print_date(self):
        now = datetime.datetime.now()
        print(f'Current date: {now.strftime("%Y-%m-%d")}')
        
    def print_time(self):
        now = datetime.datetime.now()
        print(f'Current time: {now.strftime("%H:%M:%S")}')
        
def main():
    facultyStudents = Students()
    
    while True:
        print('This console application allows you to add students,', end ='') 
        print('check their availability, and display the current date and time.')
        print('To add students enter 1, to search for a student enter 2,', end='')
        request = int(input('to display the current date enter 3, and to display the current time enter 4. '))
        
        if request == 1:
            name = input('Enter student name: ')
            facultyStudents.add_student(name)
        elif request == 2:
            name_input = input("Enter student's name if you are looking for it: ")
            
            id_input = input("Enter student's id if you are looking for it: ")
            if id_input != '':
                id_input = int(id_input)
            
            facultyStudents.print_student(id_input, name_input)
        elif request == 3:
            facultyStudents.print_date()
        elif request == 4:
            facultyStudents.print_time()
            
        exit = input('Enter something if you want to exit: ')
        if exit != '':
            break
        
    
if __name__ == "__main__":
    main()
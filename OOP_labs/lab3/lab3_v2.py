class Person():
    def __init__(self):
        self.nationality = 'world'
    
    def PrintCountryName(self):
        if self.nationality == 'Italian':
            print("This person's country is Italy.")
        elif self.nationality == 'Ukrainian':
            print("This person's country is Ukraine.")
        else:
            print("Every person has a country.")
        
    def Speak(self):
        if self.nationality == 'Italian':
            print("This person speak Italian.")
        elif self.nationality == 'Ukrainian':
            print("This person speak Ukrainian.")
        else:
            print("Every person can speak.")
        
class Italian(Person):
    def __init__(self):
        self.nationality = 'Italian'
        
class Ukrainian(Person):
    def __init__(self):
        self.nationality = 'Ukrainian'
        
def main():
    print("Hello! If you are Ukrainian, enter 1. If you are Italian, enter 2. ", end='')
    try:
        person_type = int(input("Otherwise, do not enter anything. "))
        if person_type == 1:
            person = Ukrainian()
        elif person_type == 2:
            person = Italian()
        else:
            print("You entered something, but it's not 1 or 2!")
            return
    except ValueError:
        person = Person()
        
    print()
    print_choose = input("Would you like to know some information about this person? Y/n ")
    if print_choose in ['Y', 'y']:
        person.PrintCountryName()
        person.Speak()
    
if __name__ == "__main__":
    main()
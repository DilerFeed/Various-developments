class temperatureConverter():
    
    def is_number(self, number):
        try:
            float(number)
        except ValueError:
            return False
        else:
            return True
    
    def fahrenheit2celsius(self, fahrenheit_temp=32):
        if self.is_number(fahrenheit_temp) == True:
            celsius_temp = (float(fahrenheit_temp) - 32) * (5/9)
            return celsius_temp
        else:
            print("This is not a number!")
            return None
        
    def fahrenheit2celsius_range(self, celsius_mode=False, left_border=32, right_border=32):
        if self.is_number(left_border) == True and self.is_number(right_border) == True:
            if celsius_mode == False:
                print("--------------------------------------------")
                print("Fahrenheit Temperature | Celsius Temperature")
                print("--------------------------------------------")
                for fahrenheit_temp in range(int(left_border), int(right_border)):
                    celsius_temp = (float(fahrenheit_temp) - 32) * (5/9)
                    print(f"{fahrenheit_temp}                   |                 {int(celsius_temp)}")
                    print("--------------------------------------------")
            else:
                print("--------------------------------------------")
                print("Celsius Temperature | Fahrenheit Temperature")
                print("--------------------------------------------")
                for celsius_temp in range(int(left_border), int(right_border)):
                    fahrenheit_temp = (float(celsius_temp) * (9/5)) + 32
                    print(f"{celsius_temp}                   |                 {int(fahrenheit_temp)}")
                    print("--------------------------------------------")
        else:
            print("Range boundary values must be numbers!")

            
def main():
    converter = temperatureConverter()
    
    while True:
        print("This program allows you to convert temperatures in Fahrenheit to Celsius and vice versa.")
        
        print("Select table mode (Celsius and Fahrenheit) or single number mode (Fahrenheit only).")
        mode = input("Enter 1 for table mode and 2 for single number mode or 'exit' to exit: ")
        if mode == 'exit':
            break
        elif int(mode) == 2:
            temp = input("Enter temperature in Fahrenheit: ")
            
            celsius_temp = converter.fahrenheit2celsius(temp)
            
            if celsius_temp != None:
                print(f"The value of {temp} Fahrenheit will be {celsius_temp:.3f} Celsius")
        elif int(mode) == 1:
            table_mode = input("Enter F/f for fahrenheit2celsius or C/c for celsius2fahrenheit: ")
            if table_mode == "C" or table_mode == "c":
                table_mode = True
            else:
                table_mode = False
                
            left_border = input("Enter left border of the table range: ")
            right_border = input("Enter right border of the table range: ")
            
            converter.fahrenheit2celsius_range(table_mode, left_border, right_border)
    
    
if __name__ == "__main__":
    main()
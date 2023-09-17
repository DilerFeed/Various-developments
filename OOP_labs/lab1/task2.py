import math

class Calculator():
    
    def is_number(self, number):
        try:
            float(number)
        except ValueError:
            return False
        else:
            return True
    
    def calculate_ln(self, number):
        if self.is_number(number) == True and '-' not in number \
                and number != '0':
            answer = math.log(float(number))
            return answer
        else:
            print("This is not a number or number is zero or negative!")
    
def main():
    calculator = Calculator()
    
    while True:
        print("This program allows you to calculate the value of such a formula with a given x.")
        x = input("Enter x value or 'exit' to exit: ")
        
        if x == 'exit':
            break
        ln_x = calculator.calculate_ln(x)
        
        print(f"Answer: ln({x}) = {ln_x}")
    
if __name__ == "__main__":
    main()
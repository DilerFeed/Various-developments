"""Class of Average Cost of Air Passanger Transportation Calculation List"""
class ACAPTCalculationList():
    def __init__(self, aircraft_type=[], flight=[], flight_expenses=[], 
                           passengers_num=[]):
        self.aircraft_type = aircraft_type
        self.flight = flight
        self.flight_expenses = flight_expenses
        self.passengers_num = passengers_num
        self.avg_transportation_cost = []
        
        # If the lists are not equal when initializing an instance, delete the instance
        if len(self.aircraft_type) != len(self.flight) != \
        len(self.flight_expenses) != len(self.passengers_num):
                print("Lists with different numbers of elements have been introduced! ", end='')
                print("The class instance has been deleted!")
                del(self)
            
        # If the method returns False, delete the class instance
        for f_num in range(len(flight)):
            status = self.calculate_avg_trnsp_cost(self.flight_expenses[f_num], 
                                          self.passengers_num[f_num])
            if status == False:
                del(self)
                
        # Calculate the total values immediately after initialization
        self.total = self.calculate_total()
        
        self.MAX_PAS_NUM = 250      # Maximum passengers number
        self.MIN_PAS_NUM = 50       # Minimum passengers number
        self.MIN_EXPENSES = 0       # Minimum expenses for one flight
        
        # Supported aircraft types
        self.aircraft_types = ["A320ceo", "737-800", "A321ceo", "A319ceo",
                               "737-700", "Embraer 175", "ATR-72",
                               "A320neo", "737 MAX 8", "CRJ-900"]
        
    # Method for calculation of average transportation cost and adding to the list. 
    # In case of failure, it simply returns False.
    def calculate_avg_trnsp_cost(self, flight_expenses, passengers_num):
        if passengers_num >= self.MIN_PAS_NUM and passengers_num <= self.MAX_PAS_NUM and \
        flight_expenses > self.MIN_EXPENSES:
            avg_trnsp_cost = flight_expenses / passengers_num
            self.avg_transportation_cost.append(int(avg_trnsp_cost))
            return True
        else:
            print("The data is incorrect, the process of deleting a class ", end='')
            print("instance has been requested.")
            return False
        
    # Calculates all total values at once and returns
    def calculate_total(self):
        total_flights = sum(self.flight)
        total_expenses = sum(self.flight_expenses)
        total_passengers = sum(self.passengers_num)
            
        total_avg_transportation_cost = sum(self.avg_transportation_cost)
        total_avg_transportation_cost /= len(self.avg_transportation_cost)
        
        return [total_flights, total_expenses, total_passengers,
                int(total_avg_transportation_cost)]
        
    # Adds a flight to the list and returns True if the addition is successful. 
    # Otherwise it simply returns False.
    def add_flight(self, aircraft_type="A320ceo", flight="AAA-AAA", 
                   flight_expenses=9000, passengers_num=180):
        if aircraft_type in self.aircraft_types and passengers_num >= self.MIN_PAS_NUM and \
            passengers_num <= self.MAX_PAS_NUM and len(flight) == 7 and \
            flight_expenses > self.MIN_EXPENSES:
                self.aircraft_type.append(aircraft_type)
                self.flight.append(flight)
                self.flight_expenses.append(flight_expenses)
                self.passengers_num.append(passengers_num)
                
                self.calculate_avg_trnsp_cost(flight_expenses, passengers_num)
                self.total = self.calculate_total()
                
                print("Flight added successfully!")
                return True
        else:
            print("Incorrect data entered! No data has been added. Try again.")
            return False
                
    # Prints all values for one list index in formatted form.
    def print_info(self, flight_index=0):
        print("------------------------------------")
        print(f"Aircraft type: {self.aircraft_type[flight_index]}")
        print(f"Flight: {self.flight[flight_index]}")
        print(f"Flight expenses: {self.flight_expenses[flight_index]}")
        print(f"Passengers number: {self.passengers_num[flight_index]}")
        print(f"Average transportation cost: {self.avg_transportation_cost[flight_index]}")
        print("------------------------------------")
                
    # Method for searching values in a list for different parameters
    def search(self, total=False, flight_index=-1, flight="", aircraft_type=""):
        # Displaying total statistics when prompted
        if total == True:
            print("Total statistics:")
            print("------------------------------------")
            print(f"Total flights: {self.total[0]}")
            print(f"Total expenses: {self.total[1]}")
            print(f"Total passengers: {self.total[2]}")
            print(f"Total average transportation cost: {self.total[3]}")
            print("------------------------------------")
        # Displaying a flight by its index upon request
        if flight_index != None and flight_index >= 0:
            try:
                print(f"Flight with index {flight_index} is searching!")
                self.print_info(flight_index)
            except ValueError:
                print(f"There are no flights under number {flight_index}!")
        # Displaying all flights of the same name when requested
        if len(flight) == 7:
            flight_indeces = []
            for flight_index in range(len(self.flight)):
                if self.flight[flight_index] == flight:
                    flight_indeces.append(flight_index)
            if len(flight_indeces) == 1:
                print(f"Flight {flight} found!")
                self.print_info(flight_index)
            elif len(flight_indeces) > 1:
                print(f"Several flights {flight} found!")
                for flight_index in flight_indeces:
                    self.print_info(flight_index)
            else:
                print(f"There is no such flight as {flight}!")
        # We display all flights for aircrafts of the same type upon request
        if aircraft_type in self.aircraft_types:
            flight_indeces = []
            for flight_index in range(len(self.aircraft_type)):
                if self.aircraft_type[flight_index] == aircraft_type:
                    flight_indeces.append(flight_index)
            if len(flight_indeces) == 1:
                print(f"Flight for aircraft {aircraft_type} found!")
                self.print_info(flight_index)
            elif len(flight_indeces) > 1:
                print(f"Several flights for aircrafts {aircraft_type} found!")
                for flight_index in flight_indeces:
                    self.print_info(flight_index)
            else:
                print(f"There is no flights for {aircraft_type}!")
                
def main():
    # Some values to initialize the list to not empty
    aircraft_types = ["737-800", "A321ceo", "737-700", "A320ceo", "A320ceo",
                      "ATR-72", "737 MAX 8", "A320ceo", "CRJ-900", "737-800"]
    flights = ["JJU-SEG", "SPR-TOH", "HNI-HCM", "FUK-TOH", "MLB-SYD",
               "BEJ-SHH", "LSV-LSA", "HLU-KAH", "DNV-LSV", "NYJ-LSA"]
    flight_expenses = [7850, 8340, 9230, 7989, 8130, 
                       10105, 9664, 8699, 8931, 8200]
    passengers_num = [200, 150, 180, 170, 166,
                      110, 198, 159, 80, 146]
    
    # Create an instance of a class with the given parameters
    calc_list_1 = ACAPTCalculationList(aircraft_types, flights, flight_expenses,
                                       passengers_num)
    
    # Console user interface for program management
    print("Welcome to the Shanghai airport system!")
    print("Here you can add a flight to the list or find information about any" , end='')
    print("flight at the airport.")
    while True:
        print("To add a flight, enter 1. To search for a flight or get general ", end='')
        # Since converting text without numbers to int will lead to an error, 
        # in this case we turn off the program.
        try:
            mode = int(input("statistics, enter 2. If you want to exit, enter something else. "))
            # Flight adding mode. Works until the flight is successfully added.
            if mode == 1:
                status = False
                while status == False:
                    aircraft_type = input("Enter aircraft type: ")
                    flight = input("Enter flight name (7-symbol format): ")
                    try:
                        flight_expenses = int(input("Enter flight expenses: "))
                    except ValueError:
                        print("Invalid value, try again!")
                        continue
                    try:
                        passengers_num = int(input("Enter number of passengers: "))
                    except ValueError:
                        print("Invalid value, try again!")
                        continue
                    # Use the method to add the obtained values. It also tells us about success.
                    status = calc_list_1.add_flight(aircraft_type, flight, flight_expenses,
                                                    passengers_num)
            # Flight search mode.
            # Allows you to search across different categories at one time.
            if mode == 2:
                print("If you want to know general statistics, enter something. ", end='')
                total = input("Otherwise, just press Enter. ")
                if total == None or total == '':
                    total = False
                else:
                    total = True
                    
                try:
                    print("If you want to search using flight index, enter it. ", end='')
                    flight_index = int(input("Otherwise, just press Enter. "))
                except ValueError:
                    flight_index = -1
                
                print("If you want to search using flight name, ", end='')
                flight = input("enter it (7-symbol format). Otherwise, just press Enter. ")
                
                print("If you want to search using aircraft type, ", end='')
                aircraft_type = input("enter it. Otherwise, just press Enter. ")
                
                calc_list_1.search(total, flight_index, flight, aircraft_type)
        except ValueError:
            print("We wish you a nice day! Goodbye.")
            break
            

if __name__ == "__main__":
    main()
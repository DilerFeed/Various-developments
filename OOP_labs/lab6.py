from abc import ABC, abstractmethod

class Animal(ABC):
    def __init__(self, name):
        self.name = name

class SoundProducingAnimal(ABC):
    @abstractmethod
    def makeSound(self):
        pass

class Dog(Animal, SoundProducingAnimal):
    def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed

    def makeSound(self):
        return "Woof!"

    def __eq__(self, other):
        return self.name == other.name and self.breed == other.breed

class Cat(Animal, SoundProducingAnimal):
    def __init__(self, name, color):
        super().__init__(name)
        self.color = color

    def makeSound(self):
        return "Meow!"

    def __eq__(self, other):
        return self.name == other.name and self.color == other.color

class AnimalManager:
    def __init__(self):
        self.animals = []

    def addAnimal(self, animal):
        if isinstance(animal, Animal):
            self.animals.append(animal)
        else:
            print("Invalid animal type")

    def playSounds(self):
        for animal in self.animals:
            print(f"{animal.name} says: {animal.makeSound()}")

def get_positive_integer(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

if __name__ == "__main__":
    manager = AnimalManager()

    while True:
        print("1. Add animal")
        print("2. Play sounds")
        print("3. Compare animals")
        print("0. Quit")

        try:
            action = int(input("Enter your choice: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if action == 0:
            break
        elif action == 1:
            print("1. Add dog")
            print("2. Add cat")
            print("0. Go back")

            try:
                animal_choice = int(input("Enter your choice: "))
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            if animal_choice == 0:
                continue
            elif animal_choice == 1:
                name = input("Enter the name of the dog: ")
                breed = input("Enter the breed of the dog: ")
                dog = Dog(name, breed)
                manager.addAnimal(dog)
            elif animal_choice == 2:
                name = input("Enter the name of the cat: ")
                color = input("Enter the color of the cat: ")
                cat = Cat(name, color)
                manager.addAnimal(cat)
            else:
                print("Invalid choice. Please enter 1, 2, or 0.")
        elif action == 2:
            manager.playSounds()
        elif action == 3:
            print("Enter the numbers of two animals to compare:")
            for i, animal in enumerate(manager.animals, 1):
                print(f"{i}. {animal.name} ({type(animal).__name__})")

            try:
                animal_num1 = int(input("Enter the number of the first animal: "))
                animal_num2 = int(input("Enter the number of the second animal: "))
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
                continue

            if 1 <= animal_num1 <= len(manager.animals) and 1 <= animal_num2 <= len(manager.animals):
                animal1 = manager.animals[animal_num1 - 1]
                animal2 = manager.animals[animal_num2 - 1]

                if animal1 == animal2:
                    print(f"{animal1.name} and {animal2.name} are the same.")
                else:
                    print(f"{animal1.name} and {animal2.name} are different.")
            else:
                print("Invalid animal numbers. Please enter valid numbers.")
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 0.")

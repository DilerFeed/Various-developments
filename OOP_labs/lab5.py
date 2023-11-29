from abc import ABC, abstractmethod

# Shape interface (Abstract -> cannot be created directly)
class Shape(ABC):
    # Abstract method -> must be implemented in subclasses
    @abstractmethod
    def calculateArea(self):
        pass

# A Circle class that implements the Shape interface
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def calculateArea(self):
        return 3.14 * self.radius * self.radius

# A Rectangle class that implements the Shape interface
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def calculateArea(self):
        return self.width * self.height

class ShapeManager:
    def __init__(self):
        self.shapes = []

    def addShape(self, shape):
        # Checks if the object is an instance of the Shape class
        if isinstance(shape, Shape):
            self.shapes.append(shape)
        else:
            print("Invalid shape type")

    def calculateAllAreas(self):
        return sum(shape.calculateArea() for shape in self.shapes)

    def printAllAreas(self):
        # Iterate through the list self.shapes starting from 1
        for i, shape in enumerate(self.shapes, 1):
            print(f"Area of Shape {i}: {shape.calculateArea()}")

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
    manager = ShapeManager()
    
    while True:
        print("1. Add shape")
        print("2. Calculate areas")
        print("0. Quit")

        try:
            action = int(input("Enter your choice: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if action == 0:
            break
        elif action == 1:
            print("1. Add circle")
            print("2. Add rectangle")
            print("0. Go back")

            try:
                shape_choice = int(input("Enter your choice: "))
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            if shape_choice == 0:
                continue
            elif shape_choice == 1:
                radius = get_positive_integer("Enter the radius of circle (int > 0): ")
                circle = Circle(radius)
                manager.addShape(circle)
            elif shape_choice == 2:
                width = get_positive_integer("Enter the width of rectangle (int > 0): ")
                height = get_positive_integer("Enter the height of rectangle (int > 0): ")
                rectangle = Rectangle(width, height)
                manager.addShape(rectangle)
            else:
                print("Invalid choice. Please enter 1, 2, or 0.")
        elif action == 2:
            manager.printAllAreas()
            total_area = manager.calculateAllAreas()
            print(f"Total Area of All Shapes: {total_area}")
        else:
            print("Invalid choice. Please enter 1, 2, or 0.")
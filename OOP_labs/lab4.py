class Point:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
        
    # Check 2nd object to be Point
    def check_for_point(self, other):
        if other.__class__.__name__ == 'Point':
            return True
        print("You can't compare point with other data type!")
        return False
    
    # Return True if x and y coordinates of 2 points are equal
    def __eq__(self, other):
        if self.check_for_point(other):
            if self.x == other.x and self.y == other.y:
                return True
            return False
    
    # Return True if x and y coordinates of 2 points are not equal
    def __ne__(self, other):
        if self.check_for_point(other):
            if self.x != other.x and self.y != other.y:
                return True
            return False
    
    # Return True if distance from (0;0) to point 1 is less then from (0;0) to point 2
    def __lt__(self, other):
        if self.check_for_point(other):
            self_mag = (self.x ** 2) + (self.y ** 2)
            other_mag = (other.x ** 2) + (other.y ** 2)
            return self_mag < other_mag
    
    # Return True if distance from (0;0) to point 1 is bigger then from (0;0) to point 2
    def __gt__(self, other):
        if self.check_for_point(other):
            self_mag = (self.x ** 2) + (self.y ** 2)
            other_mag = (other.x ** 2) + (other.y ** 2)
            return self_mag > other_mag
    
    def __str__(self):
        return "({0},{1})".format(self.x,self.y)
    
if __name__ == "__main__":
    p1 = Point(1,1)
    p2 = Point(-2,-3)
    p3 = Point(1,-1)
    p4 = Point(1,1)
    num = 5
    
    if p1 == p2:
        print("p1 is equal p2 (eq)")
    else:
        print("p1 is not equal p2 (eq)")
    if p1 == p4:
        print("p1 is equal p4 (eq)")
    else:
        print("p1 is not equal p4 (eq)")
        
    if p1 != p2:
        print("p1 is not equal p2 (ne)")
    else:
        print("p1 is equal p2 (ne)")
    if p1 != p4:
        print("p1 is not equal p4 (ne)")
    else:
        print("p1 is equal p4 (ne)")
        
    if p1 < p2:
        print("p1 is less then p2 (lt)")
    else:
        print("p1 is not less then p2 (lt)")
    if p1 < p3:
        print("p1 is less then p3 (lt)")
    else:
        print("p1 is not less then p3 (lt)")
        
    if p2 > p1:
        print("p2 is bigger then p1 (gt)")
    else:
        print("p2 is not bigger then p1 (gt)")
    if p1 > p3:
        print("p1 is bigger then p3 (gt)")
    else:
        print("p1 is not bigger then p3 (gt)")
        
    if p1 == num:
        print('Check')
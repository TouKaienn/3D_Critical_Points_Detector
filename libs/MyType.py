class Vec3D():
    def __init__(self,x=0,y=0,z=0):
        self.x,self.y,self.z=x,y,z
    def getValue(self):
        print(f"value of the Vec3D:x={self.x},y={self.y},z={self.z}")


class Vec2D():
    def __init__(self,x=None,y=None):
        self.x,self.y=x,y

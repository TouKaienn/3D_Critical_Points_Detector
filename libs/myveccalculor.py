from numpy.lib.twodim_base import mask_indices
from libs.MyType import *
import numpy as np

def getVecLength3D(v:Vec3D):
    length=np.sqrt(v.x**2+v.y**2+v.z**2)
    return length

def vectDot(v1:Vec3D,v2:Vec3D,default=True):#!这里改写了原来的方法，将default参数改为TRUE来使用原来的方法
    if default:
        l3=np.sqrt((v1.x-v2.x)**2+(v1.y-v2.y)**2+(v1.z-v2.z)**2)
        
        l1=getVecLength3D(v1)
        l2=getVecLength3D(v2)
        costheta=(l1*l1+l2*l2-l3*l3)/(2*l1*l2)
        result=l1*l2*costheta
        return result
    else:
        return v1.x*v2.x+v1.y*v2.y+v1.z*v2.z

def vectCross(vec1:Vec3D,vec2:Vec3D):
    res=Vec3D()
    res.x=vec1.y*vec2.z-vec2.y*vec1.z
    res.y=vec1.z*vec2.x-vec1.x*vec2.z
    res.z=vec1.x*vec2.y-vec2.x*vec1.y
    return res

def scaleVect(vec:Vec3D,sclfactor):
    sclvect=Vec3D(sclfactor*vec.x,sclfactor*vec.y,sclfactor*vec.z)
    return sclvect
    
def addVect(vec1:Vec3D,vec2:Vec3D):
    resVec=Vec3D(vec1.x+vec2.x,vec1.y+vec2.y,vec1.z+vec2.z)
    return resVec

def subtractVect(vec1:Vec3D,vec2:Vec3D):
    res=Vec3D(vec1.x-vec2.x,vec1.y-vec2.y,vec1.z-vec2.z)
    return res

def cal_eigenvalue():
    mat=np.array([[1,2*np.sqrt(3)+3j,3], [12,24,5],[-2,5,98]])
    eigen,feature=np.linalg.eig(mat)
    return eigen

def getDist2Point3D(point1:Vec3D,point2:Vec3D):
    return np.sqrt(np.power(point1.x-point2.x,2)+np.power(point1.y-point2.y,2)+np.power(point1.z-point2.z,2))

def linearInterpolation3dMoreFrac(xfrac,yfrac,zfrac,leftbottom,lefttop,rightbottom,righttop,leftbottomNextZ,lefttopNextZ,rightbottomNextZ,righttopNextZ,units):
    res_x=(1-zfrac)*((1-xfrac)*(1-yfrac)*leftbottom.x+xfrac*(1-yfrac)*rightbottom.x+(1-xfrac)*yfrac*lefttop.x+xfrac*yfrac*righttop.x)+zfrac*((1-xfrac)*(1-yfrac)*leftbottomNextZ.x+xfrac*(1-yfrac)*rightbottomNextZ.x+(1-xfrac)*yfrac*lefttopNextZ.x+xfrac*yfrac*righttopNextZ.x)
    res_y=(1-zfrac)*((1-xfrac)*(1-yfrac)*leftbottom.y+xfrac*(1-yfrac)*rightbottom.y+(1-xfrac)*yfrac*lefttop.y+xfrac*yfrac*righttop.y)+zfrac*((1-xfrac)*(1-yfrac)*leftbottomNextZ.y+xfrac*(1-yfrac)*rightbottomNextZ.y+(1-xfrac)*yfrac*lefttopNextZ.y+xfrac*yfrac*righttopNextZ.y)
    res_z=(1-zfrac)*((1-xfrac)*(1-yfrac)*leftbottom.z+xfrac*(1-yfrac)*rightbottom.z+(1-xfrac)*yfrac*lefttop.z+xfrac*yfrac*righttop.z)+zfrac*((1-xfrac)*(1-yfrac)*leftbottomNextZ.z+xfrac*(1-yfrac)*rightbottomNextZ.z+(1-xfrac)*yfrac*lefttopNextZ.z+xfrac*yfrac*righttopNextZ.z)
    res_x/=units
    res_y/=units
    res_z/=units
    return Vec3D(res_x,res_y,res_z)
    
    
if __name__ == "__main__":
    print(cal_eigenvalue()[1].imag)   
    


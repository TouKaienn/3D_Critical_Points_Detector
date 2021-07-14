<<<<<<< HEAD
import os
import numpy as np
from numpy.core.numeric import Infinity
from libs.MyType import *
from libs.myveccalculor import *
#####################################
# This utils is made by Dull_Pigeon for
# iSURE summer project.
######Bacis Setting##################
np.set_printoptions(threshold=np.inf)
GREENE_INTVL = 2
GRID_INTVL=0.05

#########Some Struct or Ref##########

class Critical_Points():
    def __init__(self, vfwidth=51, vfheight=51, vfdepth=51, vftime=1):
        self.data_file_path = '.\\Critical-Points-Utils\\data\\5cp.vec'
        self.points_data = np.fromfile(self.data_file_path, dtype='<f')
        self.vfwidth = vfwidth
        self.vfheight = vfheight
        self.vfdepth = vfdepth
        self.vftime = vftime
        self.sizeCube = self.vfwidth*self.vfheight*self.vfdepth
        self.sizeSlice = self.vfwidth*self.vfheight

        self.compute_degree_idx=[[0,1,2],[1,3,2],#front
                                 [1,5,3],[5,7,3],#right
                                 [5,4,7],[4,6,7],#back
                                 [4,0,6],[0,2,6],#left
                                 [3,7,2],[7,6,2],#top
                                 [4,5,0],[5,1,0]]#bottom

    def findCritpnts(self, timeID=0):
        v = [0]*8
        for i in range(self.vfdepth):
            for j in range(self.vfheight):
                for k in range(self.vfwidth):
                    p = i*self.sizeSlice+j*self.vfwidth+k+timeID*self.sizeCube

                    v[0] = self.points_data[p]
                    v[1] = self.points_data[p+GREENE_INTVL]
                    v[2] = self.points_data[p+self.vfwidth]
                    v[3] = self.points_data[p+self.vfwidth+GREENE_INTVL]
                    v[4] = self.points_data[p+self.sizeSlice]
                    v[5] = self.points_data[p+self.sizeSlice+GREENE_INTVL]
                    v[6] = self.points_data[p+self.vfwidth+self.sizeSlice]
                    v[7] = self.points_data[p+self.vfwidth+GREENE_INTVL+self.sizeSlice]

                    #TODO: greene's method

    def computeDegree(self, v):  # ! 注意参数是一个len=8的Vec3D列表
        if(((v[0].x >= -GRID_INTVL) or (v[1].x >= -GRID_INTVL) or v[2].x >= -GRID_INTVL or v[3].x >= -GRID_INTVL or v[4].x >= -GRID_INTVL or v[5].x >= -GRID_INTVL or v[6].x >= -GRID_INTVL or v[7].x >= -GRID_INTVL) and 
           (v[0].x <= GRID_INTVL or v[1].x <= GRID_INTVL or v[2].x <= GRID_INTVL or v[3].x <= GRID_INTVL or v[4].x <= GRID_INTVL or v[5].x <= GRID_INTVL or v[6].x <= GRID_INTVL or v[7].x <= GRID_INTVL) and 
                (v[0].y >= -GRID_INTVL or v[1].y >= -GRID_INTVL or v[2].y >= -GRID_INTVL or v[3].y >= -GRID_INTVL or v[4].y >= -GRID_INTVL or v[5].y >= -GRID_INTVL or v[6].y >= -GRID_INTVL or v[7].y >= -GRID_INTVL) and 
                (v[0].y <= GRID_INTVL or v[1].y <= GRID_INTVL or v[2].y <= GRID_INTVL or v[3].y <= GRID_INTVL or v[4].y <= GRID_INTVL or v[5].y <= GRID_INTVL or v[6].y <= GRID_INTVL or v[7].y <= GRID_INTVL) and 
                (v[0].z >= -GRID_INTVL or v[1].z >= -GRID_INTVL or v[2].z >= -GRID_INTVL or v[3].z >= -GRID_INTVL or v[4].z >= -GRID_INTVL or v[5].z >= -GRID_INTVL or v[6].z >= -GRID_INTVL or v[7].z >= -GRID_INTVL) and 
                (v[0].z <= GRID_INTVL or v[1].z <= GRID_INTVL or v[2].z <= GRID_INTVL or v[3].z <= GRID_INTVL or v[4].z <= GRID_INTVL or v[5].z <= GRID_INTVL or v[6].z <= GRID_INTVL or v[7].z <= GRID_INTVL)):
                
                tri=[0]*3
                res=0
                
                for i in range(12):
                    tri[0]=v[self.compute_degree_idx[i][0]]
                    tri[1]=v[self.compute_degree_idx[i][1]]
                    tri[2]=v[self.compute_degree_idx[i][2]]
                    


                    
                    


    def _load_path(self, data_root_path='.\\Critical-Points-Utils\\vectordata'):  # !注意更改vec文件的目录
        """[summary]
        读取vectordata的数据
        Args:
            data_root_path (str, optional): vectordata的数据路径. Defaults to '.\\Critical-Points-Utils\\vectordata'.

        Returns:
            files_path: 列表，其中包含vectordata的所有绝对地址
        """
        files = os.listdir(data_root_path)
        files_path = []
        for file in files:
            files_path.append(os.path.join(data_root_path, file))
        return files_path


if __name__ == "__main__":
    cp = Critical_Points()
    print(cp.points_data)
=======
import numpy as np
import os
import matplotlib.pyplot as plt
class Critical_Points():
    def __init__(self):
        self.files_path=self._load_data()

    def _load_data(self,file_path="C:\\Users\\lenovo\\Desktop\\暑研论文\\Notre Dame iSURE project\\critical_points\\vectordata"):
        files_name=os.listdir(file_path)
        files_path=[]
        for file in files_name:
            files_path.append(os.path.join(file_path,file))
        
        return files_path

if __name__ == "__main__":
    cp=Critical_Points()
    file_path=cp.files_path[0]

    res=np.fromfile(file_path)
    print(np.min(res))
>>>>>>> ac7fdc5442e06f093ad28be8b18361a27307f275

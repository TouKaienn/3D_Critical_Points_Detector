import os
from typing import List
import numpy as np
from libs.MyType import *
from libs.myveccalculor import *
#####################################
# This utils is made by Dull_Pigeon for
# iSURE summer project.
######Bacis Setting##################
np.set_printoptions(threshold=np.inf)
GREENE_INTVL = 2
GRID_INTVL = 0.05
GREENE_DEGREE_THRESH = 0.00001
GREENE_SCALE_THRESH = 0.0001
ALLCRITICALPOINTS_NUM=300
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
        ############################
        self.criticalPoints = [Vec3D()]*ALLCRITICALPOINTS_NUM
        self.pntNum = 0
        self.poincateIndex = [0]*ALLCRITICALPOINTS_NUM
        ############################
        self.compute_degree_idx = [[0, 1, 2], [1, 3, 2],  # front
                                   [1, 5, 3], [5, 7, 3],  # right
                                   [5, 4, 7], [4, 6, 7],  # back
                                   [4, 0, 6], [0, 2, 6],  # left
                                   [3, 7, 2], [7, 6, 2],  # top
                                   [4, 5, 0], [5, 1, 0]]  # bottom

        self.locate_point_interpolate = [[8, 0, 1], [9, 0, 2], [11, 1, 3], [10, 9, 11], [12, 2, 3], #front
                                         [22, 4, 5], [23, 4, 6], [25, 5, 7], [24, 23, 25], [26, 6, 7], #back
                                         [13, 0, 4], [14, 8, 22], [15, 1, 5], [16, 9, 23], [17, 10, 24], #middle
                                         [18, 11, 25], [19, 2, 6], [20, 12, 26], [21, 3, 7]]
        self.subcell_idx=[[0,8,9,10,13,14,16,17],#front-left-bottom
                               [9,10,2,12,16,17,19,20],#front-left-top
                               [8,1,10,11,14,15,17,18],#front-right-bottom
                               [10,11,12,3,17,18,20,21],#front-right-top
                               [13,14,16,17,4,22,23,24],#back-left-bottom
                               [16,17,19,20,23,24,6,26],#back-left-top
                               [14,15,17,18,22,5,24,25],#back-right-bottom
                               [17,18,20,21,24,25,26,7]]#back-right-top

        self.subcell_pos=[Vec3D(-0.25,-0.25,-0.25), #front-left-bottom
                          Vec3D(-0.25,0.25,-0.25), #front-left-top 
                          Vec3D(0.25,-0.25,-0.25), #front-right-bottom
                          Vec3D(0.25,0.25,-0.25),  #front-right-top
                          Vec3D(-0.25,-0.25,0.25), #front-left-bottom  
                          Vec3D(-0.25,0.25,0.25),  #front-left-top 
                          Vec3D(0.25,-0.25,0.25),  #front-right-bottom
                          Vec3D(0.25,0.25,0.25),]  #front-right-top

        self.findCritpnts()

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
                    v[7] = self.points_data[p+self.vfwidth +
                                            GREENE_INTVL+self.sizeSlice]
                    
                    # TODO: greene's method

                    degree = self.computeDegree(v)
                    if(abs(degree) > GREENE_DEGREE_THRESH):
                        pos = Vec3D(k+0.5, j+0.5, i+0.5)
                        self.locatePoint(v, pos=pos, scale=1.0, retArray=self.criticalPoints,
                                         retNum=self.pntNum, poincateIndex=self.poincateIndex)

    def computeDegree(self, v: List):  # ! 注意参数是一个len=8的Vec3D列表
        if(((v[0].x >= -GRID_INTVL) or (v[1].x >= -GRID_INTVL) or v[2].x >= -GRID_INTVL or v[3].x >= -GRID_INTVL or v[4].x >= -GRID_INTVL or v[5].x >= -GRID_INTVL or v[6].x >= -GRID_INTVL or v[7].x >= -GRID_INTVL) and
           (v[0].x <= GRID_INTVL or v[1].x <= GRID_INTVL or v[2].x <= GRID_INTVL or v[3].x <= GRID_INTVL or v[4].x <= GRID_INTVL or v[5].x <= GRID_INTVL or v[6].x <= GRID_INTVL or v[7].x <= GRID_INTVL) and
                (v[0].y >= -GRID_INTVL or v[1].y >= -GRID_INTVL or v[2].y >= -GRID_INTVL or v[3].y >= -GRID_INTVL or v[4].y >= -GRID_INTVL or v[5].y >= -GRID_INTVL or v[6].y >= -GRID_INTVL or v[7].y >= -GRID_INTVL) and
                (v[0].y <= GRID_INTVL or v[1].y <= GRID_INTVL or v[2].y <= GRID_INTVL or v[3].y <= GRID_INTVL or v[4].y <= GRID_INTVL or v[5].y <= GRID_INTVL or v[6].y <= GRID_INTVL or v[7].y <= GRID_INTVL) and
                (v[0].z >= -GRID_INTVL or v[1].z >= -GRID_INTVL or v[2].z >= -GRID_INTVL or v[3].z >= -GRID_INTVL or v[4].z >= -GRID_INTVL or v[5].z >= -GRID_INTVL or v[6].z >= -GRID_INTVL or v[7].z >= -GRID_INTVL) and
                (v[0].z <= GRID_INTVL or v[1].z <= GRID_INTVL or v[2].z <= GRID_INTVL or v[3].z <= GRID_INTVL or v[4].z <= GRID_INTVL or v[5].z <= GRID_INTVL or v[6].z <= GRID_INTVL or v[7].z <= GRID_INTVL)):

            tri = [0]*3
            a = 0

            for i in range(12):
                tri[0] = v[self.compute_degree_idx[i][0]]
                tri[1] = v[self.compute_degree_idx[i][1]]
                tri[2] = v[self.compute_degree_idx[i][2]]
                a += self.computeSolidAngle(tri)

            a /= 12.56637061  # 4pi
            return a

    def computeSolidAngle(self, v: List):
        len1 = getVecLength3D(v[0])
        len2 = getVecLength3D(v[1])
        len3 = getVecLength3D(v[2])
        t1 = np.arccos(vectDot(v[1], v[2])/(len2*len3))
        t2 = np.arccos(vectDot(v[0], v[2])/(len1*len3))
        t3 = np.arccos(vectDot(v[0], v[1])/(len1*len2))

        t = np.tan(0.25*(t1+t2+t3))
        t *= np.tan(0.25*(t1+t2-t3))
        t *= np.tan(0.25*(t2+t3-t1))
        t *= np.tan(0.25*(t3+t1-t2))

        a = np.arctan(np.sqrt(t))*4
        if(vectDot(v[0], vectCross(v[1], v[2])) < 0):
            a = -a

        return a

    def locatePoint(self, v, pos, scale, retArray, retNum, poincateIndex):
        if(scale <= GREENE_SCALE_THRESH):
            retArray[retNum[0]] = pos
            retNum[0] += 1

        subScale = scale*0.5
        c = [Vec3D()]*27

        for i in range(8):
            c[i] = v[i]
        for i in range(19):
            c[self.locate_point_interpolate[i][0]] = scaleVect(addVect(c[self.locate_point_interpolate[i][1]], c[self.locate_point_interpolate[i][2]]),0.5)
        
        degree=None
        sub=[Vec3D]*8
        for i in range(8):
            for j in range(8):
                sub[j]=c[self.subcell_idx[i][j]]
            degree=self.computeDegree(sub)
            if (abs(degree)>GREENE_DEGREE_THRESH):
                if (self.locatePoint(sub,pos=addVect(pos,scaleVect(self.subcell_pos[i],scale)),scale=subScale,retArray=self.criticalPoints,retNum=self.pntNum,poincateIndex=self.poincateIndex)):
                    poincateIndex[retNum[0]-1]=1 if (degree>GREENE_DEGREE_THRESH) else -1
                    return True
        return False

    def _load_path(self, data_root_path='.\\Critical-Points-Utils\\vectordata'):  # !注意更改vec文件的目录
        """[summary]
        读取vectordata的数据
        Args:
            data_root_path (str, optional): vectordata的数据路径. Defaults to '.\\Critical-Points-Utils\\vectordata'.

        Returns:
            files_path: 列表，其中包含vectordata的所有绝对地址
        """
        files= os.listdir(data_root_path)
        files_path= []
        for file in files:
            files_path.append(os.path.join(data_root_path, file))
        return files_path


if __name__ == "__main__":
    cp= Critical_Points()
    print(cp.criticalPoints)

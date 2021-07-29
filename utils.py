import os
from typing import List
import numpy as np
from numpy.core.defchararray import isnumeric
from libs.MyType import *
from libs.myveccalculor import *
import time
import pickle
from tqdm import tqdm
import warnings
import pandas as pd
from decimal import *
import copy
#####################################
#  * @Description: critical points detection and classification module
#  * @Author: Dai-ge
#  * @Date: 2021-7-18
#  * @LastEditors: Dai-ge
#  * @LastEditTime: 2021-7-18
######Bacis Setting##################
# np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")

GREENE_INTVL = 2
GRID_INTVL = 0.05
GREENE_DEGREE_THRESH = 0.00001#0.00001
GREENE_SCALE_THRESH = 0.0001#0.0001 
ALLCRITICALPOINTS_NUM=1000
SMALL_DIST_BOUNDARY=4
SOURCE=0x00000010
SINK=0x00000011
ATTRACT_SADDLE=0x00000012
REPEL_SADDLE=0x00000013
ATTRACT_FOCUS=0x00000001
REPEL_FOCUS=0x00000002
ATTRACT_NODE=0x00000003
REPEL_NODE=0x00000004
ATTRACT_NODE_SADDLE=0x00000005
REPEL_NODE_SADDLE=0x00000006
ATTRACT_FOCUS_SADDLE=0x00000007
REPEL_FOCUS_SADDLE=0x00000008
CENTER=0x00000009
SMALLTHRESHOLD=0.1#In the Objective-C code, Here is SMALLTHRESHOLD=0.1
MAXIMUM_EACH_TYPE=1000
CRIT_GROUP_THRESHHOLD=5
#########Some Struct or Ref##########

###TODO:Flow Field的维度十分重要
class Critical_Points():
    def __init__(self,name,vfwidth=51, vfheight=51, vfdepth=51, vftime=1,data_file_path='.\\data\\half-cylinder105.vec',Time_Saving=True,load_again=False):
        self.cp_name=name
        self.time_saving=Time_Saving
        self.data_file_path = data_file_path
        self.vfwidth = vfwidth
        self.vfheight = vfheight
        self.vfdepth = vfdepth
        self.vftime = vftime
        self.sizeCube = self.vfwidth*self.vfheight*self.vfdepth
        self.sizeSlice = self.vfwidth*self.vfheight
        self.data_path = self.init_points_data_path(datasize=self.vftime*self.sizeCube*3+100,load_again=load_again)
        # self.res=[]
        ############################
        if self.time_saving==True:
            # self.init_points_data()
            self.init_points_data_directly()
        ############################
        self.criticalPoints = []
        self.pntNum = 0
        self.poincateIndex = []
        ############################
        self.repFocus=[CRITICALPNT() for i in range(MAXIMUM_EACH_TYPE)]
        self.repSpiralSaddle=[CRITICALPNT() for i in range(MAXIMUM_EACH_TYPE)]
        self.repNode=[CRITICALPNT() for i in range(MAXIMUM_EACH_TYPE)]
        self.attrNode=[CRITICALPNT() for i in range(MAXIMUM_EACH_TYPE)]
        self.repSaddle=[CRITICALPNT() for i in range(MAXIMUM_EACH_TYPE)]
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
                          Vec3D(0.25,0.25,0.25)]  #front-right-top
        ############################
        self.jacobMatrix=[0]*9
        self.eigenValues=[0]*6
        ############################
        for time in range(self.vftime):
            print(f"The TimeID is: {time}")
            self.findCritpnts(timeID=time)
            self.classifyCripnts(timeID=time)
            # print(f"res:{max(self.res)}")

    def init_points_data_directly(self):
        self.points_data=[]
        data=np.fromfile(self.data_file_path, dtype='<f').tolist()
        print('Data Loading:')
        buffer=[]
        for item in tqdm(data):
            buffer.append(item)
            if len(buffer)%3==0:
                self.points_data.append(Vec3D(*buffer))
                buffer=[]
        for i in range(1000):
            self.points_data.append(Vec3D(0.0,0.0,0.0))

    def init_points_data(self):
        self.point_data=[]
        data=pd.read_csv(self.data_path,names=['x','y','z'])
        for item in tqdm(zip(data['x'],data['y'],data['z'])):
            self.point_data.append(Vec3D(*item))

        # cnt=0
        # for item in self.point_data:
        #     print(item.x,item.y,item.z)
        #     cnt+=1
        #     if cnt==10:
        #         break
        # print(f"dimension of data:{len(self.points_data)}")
        # print("Data saved in buffer successfully!")

    def init_points_data_path(self,datasize,load_again):

        data_path=self.data_file_path[:-3]+'csv'
        if not os.path.exists(data_path) or load_again:
            data=np.fromfile(self.data_file_path, dtype='<f')
            data_ls=data.tolist()

            length_data_ls=len(data_ls)
            print('Data Loading:')
            with open(data_path,'w') as f:
                line=[]
                for item in tqdm(data_ls):
                    line.append(str(item))
                    if len(line)==3:
                        f.write(",".join(line)+'\n')
                        line=[]
                if length_data_ls<datasize:
                    print('Data Complementing:')
                    for j in tqdm(range(datasize-length_data_ls)):
                        if j%3==0:
                            f.write('0,0,0\n')
        else:
            print("The data is exist in local machine.")
        return data_path

        
    def findCritpnts(self, timeID):
        v = [0]*8
        
        if not self.time_saving:
            print("\nData Processing:(Space Saving)")
            points_data=pd.read_csv(self.data_path,names=['x','y','z'])
            for i in tqdm(range(self.vfdepth-1)):
                for j in range(self.vfheight-1):
                    for k in range(self.vfwidth-1):

                        p = i*self.sizeSlice+j*self.vfwidth+k+timeID*self.sizeCube

                        v[0] = Vec3D(points_data.iloc[p].x,points_data.iloc[p].y,points_data.iloc[p].z)
                        v[1] = Vec3D(points_data.iloc[p+GREENE_INTVL].x,points_data.iloc[p+GREENE_INTVL].y,points_data.iloc[p+GREENE_INTVL].z)
                        v[2] = Vec3D(points_data.iloc[p+self.vfwidth].x,points_data.iloc[p+self.vfwidth].y,points_data.iloc[p+self.vfwidth].z)
                        v[3] = Vec3D(points_data.iloc[p+self.vfwidth+GREENE_INTVL].x,points_data.iloc[p+self.vfwidth+GREENE_INTVL].y,points_data.iloc[p+self.vfwidth+GREENE_INTVL].z)
                        v[4] = Vec3D(points_data.iloc[p+self.sizeSlice].x,points_data.iloc[p+self.sizeSlice].y,points_data.iloc[p+self.sizeSlice].z)
                        v[5] = Vec3D(points_data.iloc[p+self.sizeSlice+GREENE_INTVL].x,points_data.iloc[p+self.sizeSlice+GREENE_INTVL].y,points_data.iloc[p+self.sizeSlice+GREENE_INTVL].z)
                        v[6] = Vec3D(points_data.iloc[p+self.vfwidth+self.sizeSlice].x,points_data.iloc[p+self.vfwidth+self.sizeSlice].y,points_data.iloc[p+self.vfwidth+self.sizeSlice].z)
                        v[7] = Vec3D(points_data.iloc[p+self.vfwidth+GREENE_INTVL+self.sizeSlice].x,points_data.iloc[p+self.vfwidth+GREENE_INTVL+self.sizeSlice].y,points_data.iloc[p+self.vfwidth+GREENE_INTVL+self.sizeSlice].z)
                        

                        # TODO: greene's method

                        degree = self.computeDegree(v)

                        # self.res.append(p+self.vfwidth +
                                                # GREENE_INTVL+self.sizeSlice)#TODO:DELETE

                        if(abs(degree) > GREENE_DEGREE_THRESH):
                            
                                
                            pos = Vec3D(k+0.5, j+0.5, i+0.5)
                            self.locatePoint(v, pos=pos, scale=1.0, retArray=self.criticalPoints,
                                            retNum=self.pntNum, poincateIndex=self.poincateIndex)
            points_data=None
        else:
            iteration=0
            print("\nData Processing:(Time Saving)")
            for i in tqdm(range(self.vfdepth-1)):
                for j in range(self.vfheight-1):
                    for k in range(self.vfwidth-1):
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
                        iteration+=1
                        degree = self.computeDegree(v,p)

                        if iteration%1000==0:
                            with open('degree_log.txt','a') as f:
                                f.write(f"degree value at {p} is {degree} with v[0].x value:{v[0].x}\n")

                        # with open('all_degree.txt','a+') as file:
                        #     file.write(f"degree:{degree} k:{k},j:{j},i:{i}\n")

                        if(abs(degree) > GREENE_DEGREE_THRESH):
                            pos = Vec3D(k+0.5, j+0.5, i+0.5)
                            # print(f"{k},{j},{i}")
                            with open('degree.txt','a+') as f:
                                f.write(f"degree:{degree} k:{k},j:{j},i:{i}\n")

                            self.locatePoint(v, pos=pos, scale=1.0, retArray=self.criticalPoints,
                                            retNum=self.pntNum, poincateIndex=self.poincateIndex)


    def computeDegree(self, v: List,p):  # ! 注意参数是一个len=8的Vec3D列表
        if(((v[0].x >= -GRID_INTVL) or (v[1].x >= -GRID_INTVL) or ( v[2].x >= -GRID_INTVL) or (v[3].x >= -GRID_INTVL) or (v[4].x >= -GRID_INTVL) or (v[5].x >= -GRID_INTVL) or (v[6].x >= -GRID_INTVL) or (v[7].x >= -GRID_INTVL)) and
           ((v[0].x <= GRID_INTVL) or (v[1].x <= GRID_INTVL) or (v[2].x <= GRID_INTVL) or (v[3].x <= GRID_INTVL) or (v[4].x <= GRID_INTVL) or (v[5].x <= GRID_INTVL) or (v[6].x <= GRID_INTVL) or (v[7].x <= GRID_INTVL)) and
           ((v[0].y >= -GRID_INTVL) or (v[1].y >= -GRID_INTVL) or (v[2].y >= -GRID_INTVL) or (v[3].y >= -GRID_INTVL) or (v[4].y >= -GRID_INTVL) or (v[5].y >= -GRID_INTVL) or (v[6].y >= -GRID_INTVL) or (v[7].y >= -GRID_INTVL)) and
           ((v[0].y <= GRID_INTVL) or (v[1].y <= GRID_INTVL) or (v[2].y <= GRID_INTVL) or (v[3].y <= GRID_INTVL) or (v[4].y <= GRID_INTVL) or (v[5].y <= GRID_INTVL) or (v[6].y <= GRID_INTVL) or (v[7].y <= GRID_INTVL)) and
           ((v[0].z >= -GRID_INTVL) or (v[1].z >= -GRID_INTVL) or (v[2].z >= -GRID_INTVL) or (v[3].z >= -GRID_INTVL) or (v[4].z >= -GRID_INTVL) or (v[5].z >= -GRID_INTVL) or (v[6].z >= -GRID_INTVL) or (v[7].z >= -GRID_INTVL)) and
           ((v[0].z <= GRID_INTVL) or (v[1].z <= GRID_INTVL) or (v[2].z <= GRID_INTVL) or (v[3].z <= GRID_INTVL) or (v[4].z <= GRID_INTVL) or (v[5].z <= GRID_INTVL) or (v[6].z <= GRID_INTVL) or (v[7].z <= GRID_INTVL))):

            tri = [0 for i in range(3)]#[Vec3D()]*3
            a = 0.0
            tmp_ls=[]
            for i in range(12):
                tri[0] = v[self.compute_degree_idx[i][0]]
                tri[1] = v[self.compute_degree_idx[i][1]]
                tri[2] = v[self.compute_degree_idx[i][2]]
                tmp_ls.append(self.computeSolidAngle(tri,p))#!这里出现了问题，正确答案应该是-0.00033（已修复）

            a = sum(tmp_ls)/12.56637061 # 4pi

            return a
        return 0.0


    def computeSolidAngle(self, v: List,p):#This function work correctly
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
        # if p==313518:
        #     print(f"t1:{t1},t2:{t2},t3:{t3},t{t},a{a}")
        
        with open('dev_log.txt','a') as f:
            f.write(str(a)+'\n')

        return a

    def locatePoint(self, v, pos, scale, retArray, retNum, poincateIndex):
        if(scale <= GREENE_SCALE_THRESH):
            self.criticalPoints.append(pos)
            self.pntNum+= 1
            return True

        subScale = scale*0.5
        c = [Vec3D() for i in range(27)]

        for i in range(8):
            c[i] = v[i]
        for i in range(19):
            c[self.locate_point_interpolate[i][0]] = scaleVect(addVect(c[self.locate_point_interpolate[i][1]], c[self.locate_point_interpolate[i][2]]),0.5)
        
        degree=None
        sub=[Vec3D() for i in range(8)]

        for i in range(8):
            for j in range(8):
                sub[j]=c[self.subcell_idx[i][j]]

            degree=self.computeDegree(sub,p=0)

            if (abs(degree)>GREENE_DEGREE_THRESH):
                if (self.locatePoint(sub,pos=addVect(pos,scaleVect(self.subcell_pos[i],scale)),scale=subScale,retArray=self.criticalPoints,retNum=self.pntNum,poincateIndex=self.poincateIndex)):
                    self.poincateIndex.append(1 if (degree>GREENE_DEGREE_THRESH) else -1)
                    return True
        return False
################################################################################################################
    def getCritpntType3D(self,pos,poincateIndex,timeID):
        critical_type=None
        tempPos=Vec3D(pos.x,pos.y,pos.z)
        self.computeEigenValue3D(pos=tempPos,timeID=timeID)

        for i in range(0,5,2):
            for j in range(i+2,5,2):
                if(self.eigenValues[i]>self.eigenValues[j]):
                    self.eigenValues[i],self.eigenValues[j]=self.eigenValues[j],self.eigenValues[i]

        if ((self.eigenValues[0]>0) and (self.eigenValues[2]>0) and (self.eigenValues[4]>0)):
            critical_type=SOURCE
        elif((self.eigenValues[0]<0) and (self.eigenValues[2]>0) and (self.eigenValues[4]>0)):
            critical_type=REPEL_SADDLE
        elif((self.eigenValues[0]<0) and (self.eigenValues[2]<0) and (self.eigenValues[4]>0)):
            critical_type=ATTRACT_SADDLE
        elif((self.eigenValues[0]<0) and (self.eigenValues[2]<0) and (self.eigenValues[4]<0)):
            critical_type=SINK


        # TODO:test:
        # print(f"the value of the type:{critical_type}  with   the pos:x={pos.x},y={pos.y},z={pos.z}")
        # print(f"1={np.abs(self.eigenValues[1])},3={np.abs(self.eigenValues[3])},5={np.abs(self.eigenValues[5])}")
        ###########################################################
        if(critical_type==SOURCE):
            if((abs(self.eigenValues[1])<SMALLTHRESHOLD) and (abs(self.eigenValues[3])<SMALLTHRESHOLD) and (abs(self.eigenValues[5])<SMALLTHRESHOLD)):
                critical_type=REPEL_NODE
            else:
                critical_type=REPEL_FOCUS
        elif(critical_type==REPEL_SADDLE):
            if((abs(self.eigenValues[1])<SMALLTHRESHOLD) and (abs(self.eigenValues[3])<SMALLTHRESHOLD) and (abs(self.eigenValues[5])<SMALLTHRESHOLD)):
                critical_type=REPEL_NODE_SADDLE#6
            else:
                critical_type=REPEL_FOCUS_SADDLE#8
        elif(critical_type==ATTRACT_SADDLE):
            if((abs(self.eigenValues[1])<SMALLTHRESHOLD) and (abs(self.eigenValues[3])<SMALLTHRESHOLD) and (abs(self.eigenValues[5])<SMALLTHRESHOLD)):
                critical_type=ATTRACT_NODE_SADDLE
            else:
                critical_type=ATTRACT_FOCUS_SADDLE
        elif(critical_type==SINK):
            if((abs(self.eigenValues[1])<SMALLTHRESHOLD) and (abs(self.eigenValues[3])<SMALLTHRESHOLD) and (abs(self.eigenValues[5])<SMALLTHRESHOLD)):
                critical_type=ATTRACT_NODE
            else:
                critical_type=ATTRACT_FOCUS
        if((critical_type==None) and (self.eigenValues[3]!=0) and (abs(self.eigenValues[2])<SMALLTHRESHOLD)):
            critical_type=CENTER
        
        # print(f"the value of the type:{critical_type}     the pos:x={pos.x},y={pos.y},z={pos.z}") #This code is just for debugging
        # for index,item in enumerate(self.eigenValues):
        #     print(f"With eigen value e[{index}]:{item}")
        return critical_type
        
    def computeEigenValue3D(self,pos,timeID):
        jMatrix=[[Vec3D() for i in range(3)] for j in range(125)] #!注意这里维度的处理
        count=0
        points_data=pd.read_csv(self.data_path,names=['x','y','z']) if not self.time_saving else None
        jMatrix[0]=self.computeJacobianMatrix3D(points_data,pos=pos,timeID=timeID)#TODO:THIS FUNC DO NOT HAVE ANY PROBLEM

        # print(f'jMatrix[0] x:{jMatrix[0][0].x}, y:{jMatrix[0][0].y}, z:{jMatrix[0][0].z}')#TODO:TEST

        for i in range(1,3):
            ########################################---left---######################################################
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x-i,pos.y-i,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x-i,pos.y,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x-i,pos.y+i,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x-i,pos.y-i,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x-i,pos.y,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x-i,pos.y+i,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x-i,pos.y-i,pos.z+i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x-i,pos.y,pos.z+i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x-i,pos.y+i,pos.z+i),timeID=timeID)
            count+=1
            ########################################---right---######################################################
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x+i,pos.y-i,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x+i,pos.y,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x+i,pos.y+i,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x+i,pos.y-i,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x+i,pos.y,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x+i,pos.y+i,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x+i,pos.y-i,pos.z+i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x+i,pos.y,pos.z+i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x+i,pos.y+i,pos.z+i),timeID=timeID)
            count+=1
            ########################################---top---###########################################################
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x,pos.y+i,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x,pos.y+i,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x,pos.y+i,pos.z+i),timeID=timeID)
            count+=1
            ########################################---bottom---######################################################
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x,pos.y-i,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x,pos.y-i,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x,pos.y-i,pos.z+i),timeID=timeID)
            count+=1
            ########################################---front and back---######################################################
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x,pos.y,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(points_data,pos=Vec3D(pos.x,pos.y,pos.z+i),timeID=timeID)
            count+=1
        points_data=None
        for j in range(3):
            for i in range(1,count):
                jMatrix[0][j].x+=jMatrix[i][j].x
                jMatrix[0][j].y+=jMatrix[i][j].y
                jMatrix[0][j].z+=jMatrix[i][j].z
        self.jacobMatrix[0],self.jacobMatrix[1],self.jacobMatrix[2]=jMatrix[0][0].x/count,jMatrix[0][1].x/count,jMatrix[0][2].x/count
        self.jacobMatrix[3],self.jacobMatrix[4],self.jacobMatrix[5]=jMatrix[0][0].y/count,jMatrix[0][1].y/count,jMatrix[0][2].y/count
        self.jacobMatrix[6],self.jacobMatrix[7],self.jacobMatrix[8]=jMatrix[0][0].z/count,jMatrix[0][1].z/count,jMatrix[0][2].z/count
        
        # for index,item in enumerate(self.jacobMatrix):#TEST
        #     print(f'the value for {index} jacobMatrix is {item}')

        #TODO：numpy处理eigenvalue
        mat=np.array([[self.jacobMatrix[0],self.jacobMatrix[1],self.jacobMatrix[2]],
                      [self.jacobMatrix[3],self.jacobMatrix[4],self.jacobMatrix[5]],
                      [self.jacobMatrix[6],self.jacobMatrix[7],self.jacobMatrix[8]]])
        es=self.sort_es(np.linalg.eigvals(mat))

        self.eigenValues[0]=es[0].real
        self.eigenValues[1]=es[0].imag
        self.eigenValues[2]=es[1].real
        self.eigenValues[3]=es[1].imag
        self.eigenValues[4]=es[2].real
        self.eigenValues[5]=es[2].imag

    def sort_es(self,es:List):
        sorted_es=sorted(es,key=lambda x:x.real)
        return sorted_es

    def computeJacobianMatrix3D(self,points_data,pos,timeID=0):
        if not self.time_saving:
            intx,inty,intz=int(pos.x),int(pos.y),int(pos.z)
            fracx,fracy,fracz=pos.x-intx,pos.y-inty,pos.z-intz
            v=[Vec3D() for i in range(8)]
            p=timeID*self.sizeCube+intz*self.sizeSlice+inty*self.vfwidth+intx
            
            v[0]=Vec3D(points_data.iloc[p].x,points_data.iloc[p].y,points_data.iloc[p].z)
            v[1]=Vec3D(points_data.iloc[p+1].x,points_data.iloc[p+1].y,points_data.iloc[p+1].z)
            v[2]=Vec3D(points_data.iloc[p+self.vfwidth].x,points_data.iloc[p+self.vfwidth].y,points_data.iloc[p+self.vfwidth].z)
            v[3]=Vec3D(points_data.iloc[p+1+self.vfwidth].x,points_data.iloc[p+1+self.vfwidth].y,points_data.iloc[p+1+self.vfwidth].z)
            v[4]=Vec3D(points_data.iloc[p+self.sizeSlice].x,points_data.iloc[p+self.sizeSlice].y,points_data.iloc[p+self.sizeSlice].z)
            v[5]=Vec3D(points_data.iloc[p+1+self.sizeSlice].x,points_data.iloc[p+1+self.sizeSlice].y,points_data.iloc[p+1+self.sizeSlice].z)
            v[6]=Vec3D(points_data.iloc[p+self.vfwidth+self.sizeSlice].x,points_data.iloc[p+self.vfwidth+self.sizeSlice].y,points_data.iloc[p+self.vfwidth+self.sizeSlice].z)
            v[7]=Vec3D(points_data.iloc[p+1+self.vfwidth+self.sizeSlice].x,points_data.iloc[p+1+self.vfwidth+self.sizeSlice].y,points_data.iloc[p+1+self.vfwidth+self.sizeSlice].z)
            
            # self.res.append(p+1+self.vfwidth+self.sizeSlice)#DELETE
            points_data=None
            return self.interpolateJacobianMatrix3D(v,Vec3D(fracx,fracy,fracz))

        else:
            intx,inty,intz=int(pos.x),int(pos.y),int(pos.z)
            fracx,fracy,fracz=pos.x-intx,pos.y-inty,pos.z-intz
            v=[Vec3D() for i in range(8)]
            p=timeID*self.sizeCube+intz*self.sizeSlice+inty*self.vfwidth+intx
            v[0]=self.points_data[p]
            v[1]=self.points_data[p+1]
            v[2]=self.points_data[p+self.vfwidth]
            v[3]=self.points_data[p+1+self.vfwidth]
            v[4]=self.points_data[p+self.sizeSlice]
            v[5]=self.points_data[p+1+self.sizeSlice]
            v[6]=self.points_data[p+self.vfwidth+self.sizeSlice]
            v[7]=self.points_data[p+1+self.vfwidth+self.sizeSlice]
            return self.interpolateJacobianMatrix3D(v,Vec3D(fracx,fracy,fracz))


    def interpolateJacobianMatrix3D(self,v,pos):
        jacob=[Vec3D() for i in range(3)]
        pd=[Vec3D() for i in range(12)]
        pd[0]=subtractVect(v[1],v[0])
        pd[1]=subtractVect(v[5],v[4])
        pd[2]=subtractVect(v[3],v[2])
        pd[3]=subtractVect(v[7],v[6])
        pd[4]=subtractVect(v[2],v[0])
        pd[5]=subtractVect(v[3],v[1])
        pd[6]=subtractVect(v[6],v[4])
        pd[7]=subtractVect(v[7],v[5])
        pd[8]=subtractVect(v[4],v[0])
        pd[9]=subtractVect(v[5],v[1])
        pd[10]=subtractVect(v[6],v[2])
        pd[11]=subtractVect(v[7],v[3])
        
        jacob[0]=self.linearInterpolation3dFromFour(pos.z,pos.y,pd[0],pd[2],pd[1],pd[4])
        jacob[1]=self.linearInterpolation3dFromFour(pos.x,pos.z,pd[4],pd[6],pd[5],pd[7])
        jacob[2]=self.linearInterpolation3dFromFour(pos.x,pos.y,pd[8],pd[10],pd[9],pd[11])
        
        return jacob
    
    def linearInterpolation3dFromFour(self,xfrac,yfrac,leftbottom,lefttop,rightbottom,righttop):
        res=Vec3D()
        res.x=(1-xfrac)*(1-yfrac)*leftbottom.x+xfrac*(1-yfrac)*rightbottom.x+(1-xfrac)*yfrac*lefttop.x+xfrac*yfrac*righttop.x
        res.y=(1-xfrac)*(1-yfrac)*leftbottom.y+xfrac*(1-yfrac)*rightbottom.y+(1-xfrac)*yfrac*lefttop.y+xfrac*yfrac*righttop.y
        res.z=(1-xfrac)*(1-yfrac)*leftbottom.z+xfrac*(1-yfrac)*rightbottom.z+(1-xfrac)*yfrac*lefttop.z+xfrac*yfrac*righttop.z
        return res
        

    def classifyCripnts(self,timeID):#TODO:ClassifyPoint书写
        
        repFocusCount,repSpiralSaddleCount,repNodeCount,attrNodeCount,repSaddleCount=0,0,0,0,0
        for i in range(self.pntNum):
            pos1=Vec3D(self.criticalPoints[i].x,self.criticalPoints[i].y,self.criticalPoints[i].z)
            
            if ((abs(pos1.x-self.vfwidth)<SMALL_DIST_BOUNDARY) or (abs(pos1.x)<SMALL_DIST_BOUNDARY) or (abs(pos1.y)<SMALL_DIST_BOUNDARY) or (abs(pos1.y-self.vfheight)<SMALL_DIST_BOUNDARY) or (abs(pos1.z)<SMALL_DIST_BOUNDARY) or (abs(pos1.z-self.vfdepth)<SMALL_DIST_BOUNDARY)):
                continue

            critical_type=self.getCritpntType3D(pos=pos1,poincateIndex=self.poincateIndex[i],timeID=timeID)

            if((critical_type==REPEL_FOCUS) or (critical_type==ATTRACT_FOCUS)):#TODO:这里可以用append优化
                self.repFocus[repFocusCount].criticalPoint=pos1
                repFocusCount+=1
            elif((critical_type==REPEL_FOCUS_SADDLE) or (critical_type==ATTRACT_FOCUS_SADDLE)):
                self.repSpiralSaddle[repSpiralSaddleCount].criticalPoint=pos1
                repSpiralSaddleCount+=1
            elif((critical_type==REPEL_NODE)):
                self.repNode[repNodeCount].criticalPoint=pos1
                repNodeCount+=1
            elif((critical_type==ATTRACT_NODE)):
                self.attrNode[attrNodeCount].criticalPoint=pos1
                attrNodeCount+=1
            elif((critical_type==REPEL_NODE_SADDLE) or (critical_type==ATTRACT_NODE_SADDLE)):
                self.repSaddle[repSaddleCount].criticalPoint=pos1
                repSaddleCount+=1
        self.repFocus,repFocusCount=self.groupCritpnts(self.repFocus,repFocusCount,timeID)
        self.repSpiralSaddle,repSpiralSaddleCount=self.groupCritpnts(self.repSpiralSaddle,repSpiralSaddleCount,timeID)
        self.repNode,repNodeCount=self.groupCritpnts(self.repNode,repNodeCount,timeID)
        self.attrNode,attrNodeCount=self.groupCritpnts(self.attrNode,attrNodeCount,timeID)
        self.repSaddle,repSaddleCount=self.groupCritpnts(self.repSaddle,repSaddleCount,timeID)
        
        
        self.show_result_args=[('repFocus',repFocusCount,self.repFocus),('repSpiralSaddle',repSpiralSaddleCount,self.repSpiralSaddle),
              ('repNode',repNodeCount,self.repNode),('attrNode',attrNodeCount,self.attrNode),('repSaddle',repSaddleCount,self.repSaddle)]
        
    def groupCritpnts(self,onetypeCritpnt,onetypeCritNum,timeID):
        tempStorage=[[Vec3D(1e4,1e4,1e4) for i in range(300)] for j in range(300)]
        tempCount=0
        tempCount2=2

        tempStorage[0][1]=copy.deepcopy(onetypeCritpnt[0].criticalPoint)

        for i in range(onetypeCritNum):
            if (onetypeCritpnt[i].criticalPoint.x!=-100):

                tempStorage[tempCount][1]=copy.deepcopy(onetypeCritpnt[i].criticalPoint)
                tempCount2=2

                for j in range(i+1,onetypeCritNum):
                    if(getDist2Point3D(onetypeCritpnt[i].criticalPoint,onetypeCritpnt[j].criticalPoint)<CRIT_GROUP_THRESHHOLD):
                        tempStorage[tempCount][tempCount2]=copy.deepcopy(onetypeCritpnt[j].criticalPoint)
                        onetypeCritpnt[j].criticalPoint.x=-100
                        onetypeCritpnt[j].criticalPoint.y=-100
                        onetypeCritpnt[j].criticalPoint.z=-100
                        tempCount2+=1

                tempStorage[tempCount][0].x=tempCount2
                tempCount+=1
    
        # for i in range(30):
        #     for j in range(30):
        #         print(f"x:{tempStorage[i][j].x},y:{tempStorage[i][j].y},z:{tempStorage[i][j].z},i:{i},j:{j}")
        
        onetypeCritNum=0
        v=[Vec3D() for i in range(9)]
        for i in range(tempCount):
            minimalVelo=10000
            minimalPoint=Vec3D()
            for j in range(1,tempStorage[i][0].x):
                velo=Vec3D()
                pos=tempStorage[i][j]
                p=int(tempStorage[i][j].z)*self.sizeSlice+int(tempStorage[i][j].y)*self.vfwidth+(int(tempStorage[i][j].x))+timeID*self.sizeCube

                v[0]=self.points_data[p]
                v[1]=self.points_data[p+GREENE_INTVL]
                v[2]=self.points_data[p+self.vfwidth]
                v[3]=self.points_data[p+self.vfwidth+GREENE_INTVL]
                v[4]=self.points_data[p+self.sizeSlice]
                v[5]=self.points_data[p+GREENE_INTVL+self.sizeSlice]
                v[6]=self.points_data[p+self.vfwidth+self.sizeSlice]
                v[7]=self.points_data[p+GREENE_INTVL+self.vfwidth+self.sizeSlice]

                xfrac=pos.x-int(pos.x)
                yfrac=pos.y-int(pos.y)
                zfrac=pos.z-int(pos.z)

                velo=linearInterpolation3dMoreFrac(xfrac,yfrac,zfrac,v[0],v[2],v[1],v[3],v[4],v[6],v[5],v[7],1.0)
                length=getVecLength3D(velo)
                if(length<minimalVelo):
                    minimalVelo=length
                    minimalPoint=pos
            onetypeCritpnt[onetypeCritNum].criticalPoint=minimalPoint
            onetypeCritNum+=1
        return onetypeCritpnt,onetypeCritNum

##################################################################################################################
    def show_all_result(self):
        #####################Function Define#########################
        def show_result(critical_type_name,critical_num,critical_data):
            print('\n')
            print(f'The total number of {critical_type_name} type is:{critical_num}')
            if critical_num!=0:
                print('the detail info of this type is shown below:')
                for index in range(critical_num):
                    critical_data[index].show()
            else:
                print('The type of this critical points is zero!!!')
            print('\n')
        #######################SHOW RESULT############################
        print(f'\nthe detail info of the critical points from {self.cp_name}({self.pntNum} in total):\n')
        for index in range(self.pntNum):
            self.criticalPoints[index].getValue()
        for arg in self.show_result_args:
            show_result(*arg)

    def show_all_critical_points(self):
        print(f'\nthe detail info of the critical points from {self.cp_name}({self.pntNum} in total):\n')
        for index in range(self.pntNum):
            self.criticalPoints[index].getValue()


def load_cp_data(data_path):  # !注意更改vec文件的目录
    with open(data_path,'rb') as f:
        cp=pickle.load(f)
        # cp.show_all_result()
        return cp

def save_cp(cp,file_name='cp.pkl'):
    with open(file_name,'wb') as f:
        pickle.dump(cp,f)
    print('The critical points and its classification have been saved successfully.')
    
if __name__ == "__main__":
    args=[('5cp',51,51,51,1,'.\\data\\5cp.vec'),('tornado17',128,128,128,1,'.\\data\\tornado17.vec'),('tangaroa157',300,180,120,1,'.\\data\\tangaroa157.vec'),('supernova015',128,128,128,1,'.\\data\\supernova015.vec'),('supercurrent-450',256,128,32,1,'.\\data\\supercurrent-450.vec'),('half-cylinder105',640,240,80,1,'.\\data\\half-cylinder105.vec')]
    data=np.fromfile('.\\data\\5cp.vec',dtype='<f')

    # data=pd.read_csv('.\\data\\tornado17.csv',names=['x','y','z'])
    # for row in data.iterrows():
    #     print(row)
    # for i in range(len(args)):

    try:
        os.remove('degree.txt')
        os.remove('dev_log.txt')
        os.remove('all_degree.txt')
        os.remove('degree_log.txt')
    except:
        pass
    
    start_time=time.time()
    cp= Critical_Points(*args[-2])
    cp.show_all_result()
    
    save_cp(cp,f'{cp.cp_name}.pkl')
    end_time=time.time()
    print(f"time consuming:{end_time-start_time}s")

    cp=load_cp_data('supercurrent-450.pkl')
    print(cp.criticalPoints[0].x)
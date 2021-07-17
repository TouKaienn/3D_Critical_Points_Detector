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

#####################################
#  * @Description: critical points detection and classification module
#  * @Author: Dai-ge
#  * @Date: 2021-7-17
#  * @LastEditors: Dai-ge
#  * @LastEditTime: 2021-7-17
######Bacis Setting##################
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")
GREENE_INTVL = 2
GRID_INTVL = 0.05
GREENE_DEGREE_THRESH = 0.00001
GREENE_SCALE_THRESH = 0.0001
ALLCRITICALPOINTS_NUM=300

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

SMALLTHRESHOLD=0.1

MAXIMUM_EACH_TYPE=100
#########Some Struct or Ref##########


class Critical_Points():
    def __init__(self, vfwidth=51, vfheight=51, vfdepth=51, vftime=1):
        self.data_file_path = '.\\Critical-Points-Utils\\data\\5cp.vec'
        self.points_data = self.init_points_data(datasize=100*100*20*48)
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
        self.findCritpnts()
        self.classifyCripnts()

    def init_points_data(self,datasize):

        data=np.fromfile(self.data_file_path, dtype='<f')
        res=[Vec3D(0,0,0)]*datasize
        buffer=[]

        print('Data Loading:')
        count=0
        for item in tqdm(data):
            buffer.append(item)

            if len(buffer)==3:
                res[count]=Vec3D(buffer[0],buffer[1],buffer[2])
                count+=1
                buffer=[]

        return res

    def findCritpnts(self, timeID=0):
        v = [0]*8


        print("\nData Processing:")

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

                    degree = self.computeDegree(v)


                    if(abs(degree) > GREENE_DEGREE_THRESH):

                        pos = Vec3D(k+0.5, j+0.5, i+0.5)
                        self.locatePoint(v, pos=pos, scale=1.0, retArray=self.criticalPoints,
                                         retNum=self.pntNum, poincateIndex=self.poincateIndex)

    def computeDegree(self, v: List):  # ! 注意参数是一个len=8的Vec3D列表
        if(((v[0].x >= -GRID_INTVL) or (v[1].x >= -GRID_INTVL) or ( v[2].x >= -GRID_INTVL) or (v[3].x >= -GRID_INTVL) or (v[4].x >= -GRID_INTVL) or (v[5].x >= -GRID_INTVL) or (v[6].x >= -GRID_INTVL) or (v[7].x >= -GRID_INTVL)) and
           ((v[0].x <= GRID_INTVL) or (v[1].x <= GRID_INTVL) or (v[2].x <= GRID_INTVL) or (v[3].x <= GRID_INTVL) or (v[4].x <= GRID_INTVL) or (v[5].x <= GRID_INTVL) or (v[6].x <= GRID_INTVL) or (v[7].x <= GRID_INTVL)) and
                ((v[0].y >= -GRID_INTVL) or (v[1].y >= -GRID_INTVL) or (v[2].y >= -GRID_INTVL) or (v[3].y >= -GRID_INTVL) or (v[4].y >= -GRID_INTVL) or (v[5].y >= -GRID_INTVL) or (v[6].y >= -GRID_INTVL) or (v[7].y >= -GRID_INTVL)) and
                ((v[0].y <= GRID_INTVL) or (v[1].y <= GRID_INTVL) or (v[2].y <= GRID_INTVL) or (v[3].y <= GRID_INTVL) or (v[4].y <= GRID_INTVL) or (v[5].y <= GRID_INTVL) or (v[6].y <= GRID_INTVL) or (v[7].y <= GRID_INTVL)) and
                ((v[0].z >= -GRID_INTVL) or (v[1].z >= -GRID_INTVL) or (v[2].z >= -GRID_INTVL) or (v[3].z >= -GRID_INTVL) or (v[4].z >= -GRID_INTVL) or (v[5].z >= -GRID_INTVL) or (v[6].z >= -GRID_INTVL) or (v[7].z >= -GRID_INTVL)) and
                ((v[0].z <= GRID_INTVL) or (v[1].z <= GRID_INTVL) or (v[2].z <= GRID_INTVL) or (v[3].z <= GRID_INTVL) or (v[4].z <= GRID_INTVL) or (v[5].z <= GRID_INTVL) or (v[6].z <= GRID_INTVL) or (v[7].z <= GRID_INTVL))):

            tri = [Vec3D(0,0,0)]*3
            a = 0

            for i in range(12):
                tri[0] = v[self.compute_degree_idx[i][0]]
                tri[1] = v[self.compute_degree_idx[i][1]]
                tri[2] = v[self.compute_degree_idx[i][2]]
                a += self.computeSolidAngle(tri)

            a = a/12.56637061  # 4pi

            return a
        return 0

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
            self.criticalPoints[self.pntNum] = pos
            self.pntNum+= 1
            return True

        subScale = scale*0.5
        c = [Vec3D()]*27

        for i in range(8):
            c[i] = v[i]
        for i in range(19):
            c[self.locate_point_interpolate[i][0]] = scaleVect(addVect(c[self.locate_point_interpolate[i][1]], \
                 c[self.locate_point_interpolate[i][2]]),0.5)
        
        degree=None
        sub=[Vec3D()]*8

        for i in range(8):
            for j in range(8):
                sub[j]=c[self.subcell_idx[i][j]]

            degree=self.computeDegree(sub)

            if (abs(degree)>GREENE_DEGREE_THRESH):
                if (self.locatePoint(sub,pos=addVect(pos,scaleVect(self.subcell_pos[i],scale)),scale=subScale,retArray=self.criticalPoints,retNum=self.pntNum,poincateIndex=self.poincateIndex)):
                    self.poincateIndex[self.pntNum-1]=1 if (degree>GREENE_DEGREE_THRESH) else -1
                    return True
        return False
################################################################################################################
    def getCritpntType3D(self,pos,poincateIndex,timeID=0):
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

        if(critical_type==SOURCE):
            if((abs(self.eigenValues[1])<SMALLTHRESHOLD) and (abs(self.eigenValues[3])<SMALLTHRESHOLD) and (abs(self.eigenValues[5])<SMALLTHRESHOLD)):
                critical_type=REPEL_NODE
            else:
                critical_type=REPEL_FOCUS
        elif(critical_type==REPEL_SADDLE):
            if((abs(self.eigenValues[1])<SMALLTHRESHOLD) and (abs(self.eigenValues[3])<SMALLTHRESHOLD) and (abs(self.eigenValues[5])<SMALLTHRESHOLD)):
                critical_type=REPEL_NODE_SADDLE
            else:
                critical_type=REPEL_FOCUS_SADDLE
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
        return critical_type
        
    def computeEigenValue3D(self,pos,timeID=0):
        jMatrix=[[Vec3D() for i in range(3)] for j in range(125)] #!注意这里维度的处理
        count=0
        jMatrix[0]=self.computeJacobianMatrix3D(pos=pos,timeID=0)
        for i in range(1,3):
            ########################################---left---######################################################
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x-i,pos.y-i,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x-i,pos.y,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x-i,pos.y+i,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x-i,pos.y-i,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x-i,pos.y,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x-i,pos.y+i,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x-i,pos.y-i,pos.z+i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x-i,pos.y,pos.z+i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x-i,pos.y+i,pos.z+i),timeID=timeID)
            count+=1
            ########################################---right---######################################################
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x+i,pos.y-i,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x+i,pos.y,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x+i,pos.y+i,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x+i,pos.y-i,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x+i,pos.y,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x+i,pos.y+i,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x+i,pos.y-i,pos.z+i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x+i,pos.y,pos.z+i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x+i,pos.y+i,pos.z+i),timeID=timeID)
            count+=1
            ########################################---top---###########################################################
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x,pos.y+i,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x,pos.y+i,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x,pos.y+i,pos.z+i),timeID=timeID)
            count+=1
            ########################################---bottom---######################################################
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x,pos.y-i,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x,pos.y-i,pos.z),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x,pos.y-i,pos.z+i),timeID=timeID)
            count+=1
            ########################################---front and back---######################################################
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x,pos.y,pos.z-i),timeID=timeID)
            count+=1
            jMatrix[count]=self.computeJacobianMatrix3D(pos=Vec3D(pos.x,pos.y,pos.z+i),timeID=timeID)
            count+=1
        
        for j in range(3):
            for i in range(1,count):
                jMatrix[0][j].x=jMatrix[i][j].x
                jMatrix[0][j].y=jMatrix[i][j].y
                jMatrix[0][j].z=jMatrix[i][j].z
        self.jacobMatrix[0],self.jacobMatrix[1],self.jacobMatrix[2]=jMatrix[0][0].x/count,jMatrix[0][1].x/count,jMatrix[0][2].x/count
        self.jacobMatrix[3],self.jacobMatrix[4],self.jacobMatrix[5]=jMatrix[0][0].y/count,jMatrix[0][1].y/count,jMatrix[0][2].y/count
        self.jacobMatrix[6],self.jacobMatrix[7],self.jacobMatrix[8]=jMatrix[0][0].z/count,jMatrix[0][1].z/count,jMatrix[0][2].z/count
        
        #TODO：numpy处理eigenvalue
        mat=np.array([[self.jacobMatrix[0],self.jacobMatrix[1],self.jacobMatrix[2]],
                      [self.jacobMatrix[3],self.jacobMatrix[4],self.jacobMatrix[5]],
                      [self.jacobMatrix[6],self.jacobMatrix[7],self.jacobMatrix[8]]])
        es,_=np.linalg.eig(mat)
        self.eigenValues[0]=es[0].real
        self.eigenValues[1]=es[0].imag
        self.eigenValues[2]=es[1].real
        self.eigenValues[3]=es[1].imag
        self.eigenValues[4]=es[2].real
        self.eigenValues[5]=es[2].imag


    def computeJacobianMatrix3D(self,pos,timeID=0):
        intx,inty,intz=int(pos.x),int(pos.y),int(pos.z)
        fracx,fracy,fracz=pos.x-intx,pos.y-inty,pos.z-intz
        v=[Vec3D()]*8
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
        jacob=[Vec3D()]*3#!: Jacob可能是一个公共变量
        pd=[Vec3D()]*12
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
        
        jacob[0]=self.linearInterpolation3dFromFour(pos.x,pos.y,pd[0],pd[2],pd[1],pd[4])
        jacob[1]=self.linearInterpolation3dFromFour(pos.x,pos.y,pd[4],pd[6],pd[5],pd[7])
        jacob[2]=self.linearInterpolation3dFromFour(pos.x,pos.y,pd[8],pd[10],pd[9],pd[11])
        
        return jacob
    
    def linearInterpolation3dFromFour(self,xfrac,yfrac,leftbottom,lefttop,rightbottom,righttop):
        res=Vec3D()
        res.x=(1-xfrac)*(1-yfrac)*leftbottom.x+xfrac*(1-yfrac)*rightbottom.x+(1-xfrac)*yfrac*lefttop.x+xfrac*yfrac*righttop.x
        res.y=(1-xfrac)*(1-yfrac)*leftbottom.y+xfrac*(1-yfrac)*rightbottom.y+(1-xfrac)*yfrac*lefttop.y+xfrac*yfrac*righttop.y
        res.z=(1-xfrac)*(1-yfrac)*leftbottom.z+xfrac*(1-yfrac)*rightbottom.z+(1-xfrac)*yfrac*lefttop.z+xfrac*yfrac*righttop.z
        return res
        

    def classifyCripnts(self,timeID=0):#TODO:ClassifyPoint书写
        
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
        
        #######################SHOW RESULT###########
        print('\nthe detail info of the critical points:\n')
        for index in range(self.pntNum):
            self.criticalPoints[index].getValue()
        # print(f"The total number of each type is:\nrepFocus:{repFocusCount+1}\nrepSpiralSaddle:{repSaddleCount+1}\nrepNode:{repNodeCount+1}\nattrNode{attrNodeCount+1}\nrepSaddle:{repSaddleCount+1}\n")
        args=[('repFocus',repFocusCount,self.repFocus),('repSpiralSaddle',repSpiralSaddleCount,self.repSpiralSaddle),
              ('repNode',repNodeCount,self.repNode),('attrNode',attrNodeCount,self.attrNode),('repSaddle',repSaddleCount,self.repSaddle)]
        for arg in args:
            self.show_result(*arg)

            
##################################################################################################################
    def show_result(self,critical_type_name,critical_num,critical_data):
        print('\n')
        print(f'The total number of {critical_type_name} type is:{critical_num}')
        if critical_num!=0:
            print('the detail info of this type is shown below:')
            for index in range(critical_num):
                critical_data[index].show()
        else:
            print('The type of this critical points is zero!!!')
        print('\n')
        
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

def save_cp(cp,file_name='cp.pkl'):
    with open(file_name,'wb') as f:
        pickle.dump(cp,f)
    
if __name__ == "__main__":
    start_time=time.time()
    cp= Critical_Points()
    end_time=time.time()    
    print(f'time_consuming:{(end_time-start_time)/60}min')





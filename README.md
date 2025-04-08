# 3D_Critical_Points_Detector

Critical Points Calculator Modules for detecting and analysing different types of critical points in the flow field. 



## Installation

To install and use this utils, you might need the following Python library below:

```bash
  numpy >= 1.21
  tqdm >= 4.61.2
```
After having these library, you could download this utils by:

```bash
git clone https://github.com/Dai-ge/3D_Critical_Points_Detector.git
```
The result should be like the following below:
```bash
Data Loading:
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 397953/397953 [00:00<00:00, 1033770.30it/s] 

Data Processing:
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 14.71it/s] 
time_consuming:3.9359123706817627s
The critical points and its classification have been saved successfully.
```
## API Reference

#### import the utils and create critical points detector
```python
  from utils.py import *
  cp=Critical_Points(vfwidth=51, vfheight=51, vfdepth=51, vftime=1,data_file_path='.\\data\\5cp.vec') 
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `vfwidth` | `int` | **Option**. the width of the 3D flow field, default=51 |
| `vfheight` | `int` | **Option**. the height of the 3D flow field, default=51 |
| `vfdepth` | `int` | **Option**. the depth of the 3D flow field, default=51 |
| `vftime` | `int` | **Option**. the timeID of the 3D flow field, default=1 |
| `data_file_path` | `string` | **Option**. the data path of the flow field data, default='.\\data\\5cp.vec' |

#### GET 3D flow filed data

```python
  data=cp.points_data
```
#### GET all critical points objects

```python
  critical_points=cp.criticalPoints
```
#### GET critical points number

```python
  critical_points_num=cp.pntNum
```
#### show the detection and classification result
```python
  cp.show_all_result()
```

## Two important objects

**Vec3D**

```python
  class Vec3D():
    def __init__(self,x=0,y=0,z=0):
        self.x,self.y,self.z=x,y,z
    def getValue(self):
        print(f"value of the Vec3D:x={self.x},y={self.y},z={self.z}")
```
For the critical points detector, its attribute criticalPoints is a list consistent with Vec3D()
```python
  cp.criticalPoints = [Vec3D()]*ALLCRITICALPOINTS_NUM
```
**CRITICALPNT**
```python
  class CRITICALPNT():
    def __init__(self):
        self.critical_type=None
        self.criticalPoint=Vec3D()

        self.tmplateSeeds=[Vec3D()]*32
        self.tmplateSeedsNum=None

    def show(self):
        print(f"this critical value is:x:{self.criticalPoint.x},y={self.criticalPoint.y},z={self.criticalPoint.z}.")
```
For the critical points detector, its attribute repFocus, repSpiralSaddle, repNode, attrNode and repSaddle is a list consistent with CRITICALPNT.

```python
  cp.repFocus=[CRITICALPNT() for i in range(MAXIMUM_EACH_TYPE)]
  cp.repSpiralSaddle=[CRITICALPNT() for i in range(MAXIMUM_EACH_TYPE)]
  cp.repNode=[CRITICALPNT() for i in range(MAXIMUM_EACH_TYPE)]
  cp.attrNode=[CRITICALPNT() for i in range(MAXIMUM_EACH_TYPE)]
  cp.repSaddle=[CRITICALPNT() for i in range(MAXIMUM_EACH_TYPE)]
```

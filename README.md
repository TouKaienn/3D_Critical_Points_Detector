# 3D_Critical_Points_Detector
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
Critical Points Calculator Modules for 2021 ISURE Program

The aim of this project is to re-build a previous critical-points calculator by Python. To check the previous project, you could look up the paper 《FlowVisual: A Visualization App for Teaching and Understanding 3D Flow Field Concepts》.




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
#### Reload the critical points detector
```python
  cp=load_cp_data(data_path='cp.pkl')
```
| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `data_path` | `string` | **Option**. the data path of the saved critical points detector, default='cp.pkl' |

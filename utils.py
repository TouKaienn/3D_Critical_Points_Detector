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
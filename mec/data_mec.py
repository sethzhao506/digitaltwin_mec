import math
import numpy as np

class DataMEC:
    def __init__(self, path):
        self.data_path = path
        self.data = self.loadData(path)
        
    def loadData(self, path):
        data = np.load(path, allow_pickle=True).item()
        print("Data Loaded! Information:")
        for key in data:
            print(f"{key}: shape {data[key].shape}")
        return data
    
    def MEC(self):
        # get sorted table, sorted by sum value of the input vector
        table = []
        for row in range(len(self.data["class"])):
            sum_v = np.sum(self.data["img"][row]) + np.sum(self.data["points"][row]) # The input has two parts: rgb image and point cloud
            label = self.data["class"][row]
            table.append((sum_v, label))
        sortedtable = sorted(table, key=lambda x: x[0])
        # calculate threshold based on class distribution in the sorted table
        num_classes = 8 # devide classes by rotation, 8 classes in total
        thresholds = np.zeros(num_classes) # each class needs a threshold
        current_class = -1
        for row in range(len(sortedtable)):
            if sortedtable[row][1] != current_class:
                thresholds[sortedtable[row][1]] += 1
                current_class = sortedtable[row][1]
        # calculate final mec
        mec = 0
        _, H, W, C = self.data["img"].shape
        _, N, M = self.data["points"].shape
        d = H*W*C + N*M # The dimension of input vector: the dimension of the combination of flattened img and points vector 
        for thres in thresholds:
            minthreshs = math.log2(thres + 1)
            mec += (1/num_classes) * ((minthreshs * d + 1) + (minthreshs + 1))
        return mec
        
        
if __name__ == "__main__":
    data_path = "../dataset/Data/Train.npy"
    dt_mec = DataMEC(data_path)
    mec = dt_mec.MEC()
    print(f"MEC of dataset {data_path}: {mec}")
        
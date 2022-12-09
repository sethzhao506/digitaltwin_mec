import math
import random
import numpy as np
import matplotlib.pyplot as plt

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
    
    def DataDistribution(self, plot=False):
        num_classes = 8
        distribute = np.zeros(num_classes)
        for c in self.data["class"]:
            distribute[c] += 1
        print("Data distribution: ", distribute)
        if plot:
            left = list(range(num_classes))
            height = distribute
            plt.bar(left, height, color=['skyblue', 'orange'], width=0.8)
            plt.xlabel('class')
            plt.ylabel('Appear Times')
            plt.show()
        return distribute
            
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
        threshold = 0
        current_class = -1
        for row in range(len(sortedtable)):
            if sortedtable[row][1] != current_class:
                threshold += 1
                current_class = sortedtable[row][1]
        # calculate final mec
        mec = 0
        _, H, W, C = self.data["img"].shape
        _, N, M = self.data["points"].shape
        d = H*W*C + N*M # The dimension of input vector: the dimension of the combination of flattened img and points vector 
        minthreshs = math.log2(threshold + 1)
        mec = minthreshs * (d + 1) + (minthreshs + 1) * num_classes # mec for multi-classes
        return mec
    
    def MEC_Progression(self, plot=False):
        idx_list = list(range(len(self.data["class"])))
        percentage_list = list(range(0, 100, 2))
        mecs = []
        for percentage in percentage_list:
            percentage /= 100.0
            sampled_idx_list = random.sample(idx_list, round(len(idx_list)*percentage))
            # get sorted table, sorted by sum value of the input vector
            table = []
            for row in sampled_idx_list:
                sum_v = np.sum(self.data["img"][row]) + np.sum(self.data["points"][row]) # The input has two parts: rgb image and point cloud
                label = self.data["class"][row]
                table.append((sum_v, label))
            sortedtable = sorted(table, key=lambda x: x[0])
            # calculate threshold based on class distribution in the sorted table
            num_classes = 8 # devide classes by rotation, 8 classes in total
            threshold = 0
            current_class = -1
            for row in range(len(sortedtable)):
                if sortedtable[row][1] != current_class:
                    threshold += 1
                    current_class = sortedtable[row][1]
            # calculate final mec
            mec = 0
            _, H, W, C = self.data["img"].shape
            _, N, M = self.data["points"].shape
            d = H*W*C + N*M # The dimension of input vector: the dimension of the combination of flattened img and points vector 
            minthreshs = math.log2(threshold + 1)
            mec = minthreshs * (d + 1) + (minthreshs + 1) * num_classes # mec for multi-classes
            mecs.append(mec)
        if plot:
            plt.plot(percentage_list, mecs, label="Ideal Machine Learner")
            plt.xlabel('% Training Data')
            plt.ylabel('MEC / bits')
            plt.title('Capacity Progression for Training Data')
            plt.legend()
            plt.show()
            
  
if __name__ == "__main__":
    data_path = "../dataset/Data/Train.npy"
    dt_mec = DataMEC(data_path)
    mec = dt_mec.MEC()
    print(f"MEC of dataset {data_path}: {mec}")
    
    dt_mec.DataDistribution(plot=True)
    dt_mec.MEC_Progression(plot=True)
        
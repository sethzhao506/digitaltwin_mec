import os
import json
import csv
import tqdm
import numpy as np
import numpy.ma as ma
import cv2
from PIL import Image

from transformations import euler_from_matrix

Borderlist = [-1] + list(range(20, 1320, 20))

def binary_search(sorted_list, target):
    l = 0
    r = len(sorted_list)-1
    while l!=r:
        mid = (l+r)>>1
        if sorted_list[mid] > target:
            r = mid
        elif sorted_list[mid] < target:
            l = mid + 1
        else:
            return mid
    return l

class DatasetBuilder:
    def __init__(self, root="./DTTD_Dataset", target_obj=2, img_size=47, point_size=300, sample_rate=1.0):
        self.data_root = root
        self.target_obj = target_obj
        self.sample_rate = sample_rate
        self.prefix = "data"
        self.pt_num = point_size
        self.img_size = img_size
        self.xmap = np.array([[j for i in range(1280)] for j in range(720)])
        self.ymap = np.array([[i for i in range(1280)] for j in range(720)])
        self._loadDataList()
        
    def _loadDataList(self):
        train_list_path = os.path.join(self.data_root, "train_data_list.txt")
        test_list_path = os.path.join(self.data_root, "test_data_list.txt")
        # load training list
        with open(train_list_path, 'r') as f:
            self.train_list = [l for l in f.read().splitlines() if not l.startswith("synthetic")]
        # load testing list
        with open(test_list_path, 'r') as f:
            self.test_list = [l for l in f.read().splitlines() if not l.startswith("synthetic")]
            
    def _getBoundingBox(self, mask, border_list):
        img_h, img_w = mask.shape
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmax += 1
        cmax += 1
        size = border_list[binary_search(border_list, max(rmax - rmin, cmax-cmin))]
        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
        rmin = center[0] - int(size / 2)
        rmax = center[0] + int(size / 2)
        cmin = center[1] - int(size / 2)
        cmax = center[1] + int(size / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > img_h:
            delt = rmax - img_h
            rmax = img_h
            rmin -= delt
        if cmax > img_w:
            delt = cmax - img_w
            cmax = img_w
            cmin -= delt
        return rmin, rmax, cmin, cmax
    
    def _classFromRotation(self, ax, ay, az):
        out = 0
        if ax > 0:
            out += 1
        if ay > 0:
            out += 2
        if az > 0:
            out += 4
        return out
            
    def _build(self, path_list):
        imgs = []
        points = []
        labels = []
        for path in tqdm.tqdm(path_list):
            try:
                # load meta data
                with open(os.path.join(self.data_root, self.prefix, f"{path}_meta.json"), "r") as f:
                    meta = json.load(f) # dict_keys(['objects', 'object_poses', 'intrinsic', 'distortion'])
                if not self.target_obj in meta['objects']:
                    continue
                
                # load raw data
                img = np.array(Image.open(os.path.join(self.data_root, self.prefix, f"{path}_color.jpg")))   # PIL, size: (1280, 720, 3)
                depth = np.array(Image.open(os.path.join(self.data_root, self.prefix, f"{path}_depth.png")), dtype=np.uint16) # shape: (720, 1280)
                label = np.array(Image.open(os.path.join(self.data_root, self.prefix, f"{path}_label.png"))) # shape: (720, 1280)
                
                # get mask and bbox
                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(label, self.target_obj))
                mask = mask_label * mask_depth # consider both label (where = objs[obj_idx]) and valid depth (where > 0)
                rmin, rmax, cmin, cmax = self._getBoundingBox(mask_label, Borderlist)
                
                # set sample points (2D) on depth/point cloud
                sample = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0] # non-zero positions on flattened mask, 1-D array
                if len(sample) >= self.pt_num:
                    sample = np.array(sorted(np.random.choice(sample, self.pt_num))) # randomly choose pt_num points (idx)
                elif len(sample) == 0:
                    sample = np.pad(sample, (0, self.pt_num - len(sample)), 'constant')
                else:
                    sample = np.pad(sample, (0, self.pt_num - len(sample)), 'wrap')
                    
                # crop image, depth and xy map with bbox, take the sample points
                img_crop = img[rmin:rmax, cmin:cmax, :]
                depth_crop = depth[rmin:rmax, cmin:cmax].flatten()[sample][:, np.newaxis].astype(np.float32) # (pt_num, )
                xmap_crop = self.xmap[rmin:rmax, cmin:cmax].flatten()[sample][:, np.newaxis].astype(np.float32) # (pt_num, ) store y for sample points
                ymap_crop = self.ymap[rmin:rmax, cmin:cmax].flatten()[sample][:, np.newaxis].astype(np.float32) # (pt_num, ) store x for sample points
                
                # set camera focus and center
                cam = np.array(meta['intrinsic'])
                cam_cx, cam_cy, cam_fx, cam_fy =  cam[0][2], cam[1][2], cam[0][0], cam[1][1]
                
                # get point cloud [[px, py, pz], ...]
                cam_scale = 1000 # uint16 * 1000
                pz = depth_crop / cam_scale
                px = (ymap_crop - cam_cx) * pz / cam_fx
                py = (xmap_crop - cam_cy) * pz / cam_fy
                point_cloud = np.concatenate((px, py, pz), axis=1) # (pt_num, 3) store XYZ point cloud value for sample points
                
                # resize image
                img_out = cv2.resize(img_crop, (self.img_size, self.img_size))
                
                # process transformation
                # get ground truth rotation and translation
                R_gt = np.array(meta['object_poses'][str(self.target_obj)])[0:3, 0:3] # (3, 3)
                ax, ay, az = euler_from_matrix(R_gt)
                cls_label = self._classFromRotation(ax, ay, az)
                
                imgs.append(img_out)
                points.append(point_cloud)
                labels.append(cls_label)
            except Exception as e:
                print("[Warning] catch an error: ", e)
        data = {"img": np.array(imgs), "points": np.array(points), "class": np.array(labels)}
        return data
            
    def buildTrain(self, save_dir="./Data/"):
        data = self._build(self.train_list)
        np.save(os.path.join(save_dir, "Train.npy"), data)
            
    def buildTest(self, save_dir="./Data/"):
        data = self._build(self.test_list)
        np.save(os.path.join(save_dir, "Test.npy"), data)
            
            
if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.buildTrain()
    builder.buildTest()
    
    data = np.load("./Data/Train.npy", allow_pickle=True)
    for key in data.item():
        print(key, data.item()[key].shape)
            
            
            
            
            
            
            
            
        
        
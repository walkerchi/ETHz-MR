"""
    GaussProcessFilter
"""
from typing import Optional
import numpy as np
import open3d as o3d
from sklearn.linear_model import LinearRegression


class HandLinearRegressionFilter:
    def __init__(self,window=32, clip=(-0.01,0.01)):
        self.window = window
        self.clip   = clip
        self.reset()
    def __len__(self):
        return len(self.ts)
    def __call__(self, hand:np.ndarray, t:np.uint64)->np.ndarray:
        xyz = hand[0,:3]
        if len(self.ts)  > 0:
            dxyz= xyz - self.xyz
        else:
            dxyz= np.zeros(3)
        dxyz_clip= np.clip(dxyz, *self.clip)
        if self.is_burnin:
            model = LinearRegression()
            model.fit(np.array(self.ts)[:,None], np.stack(self.dxyzs))
            p   = model.predict(np.array([[t]]))[0]
            # print(p)
            hand[:,:3] += (p-dxyz)
        self.add_point(dxyz_clip,t)
        self.xyz = xyz
        return hand
    @property
    def is_burnin(self):
        return len(self) >= self.window
    def add_point(self, dxyz:np.ndarray, t:np.uint64):
        dxyz = dxyz
        self.dxyzs.append(dxyz)
        self.ts.append(t)
        if self.window is not None and len(self) > self.window:
            self.dxyzs.pop(0)
            self.ts.pop(0)
    def reset(self):
        self.dxyzs = []
        self.xyz = None
        self.ts = []
        self.shape = None


class HandStaticFilter:
    def __init__(self, window_size:int=16,alpha=0.05):
        self.window_size = window_size
        self.window = []
        self.alpha  = alpha
    def __call__(self, hand:np.ndarray)->bool:
        self.window.append(np.linalg.norm(hand[:,:3]) if hand is not None else 0)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        array = np.array(self.window)
        num_none = (array == 0).sum()
        std      = array[array!=0].std()
        if std > self.alpha or num_none >= 1:
            return False
        return True


class PointCloudClusterFilter:
    def __init__(self, 
                std_scale:float=1.0,
                window_size:int=16, 
                nb_neighbors=16, 
                std_ratio=0.8, 
                eps=0.013, 
                min_points=64):
        self.std_scale   = std_scale
        self.window_size = window_size
        self.nb_neighbors= nb_neighbors
        self.std_ratio   = std_ratio
        self.eps         = eps 
        self.min_points  = min_points 
        self.xyzs        = []
        self.drs         = [0.0]
        self.rs          = []
    def __len__(self):
        return len(self.xyzs)
    def __call__(self, pcd:o3d.geometry.PointCloud, hands:np.ndarray)->Optional[o3d.geometry.PointCloud]:
        _, index = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors,std_ratio=self.std_ratio)
        pcd      = pcd.select_by_index(index)
        points   = np.array(pcd.points)
        labels   = np.array(pcd.cluster_dbscan(eps=self.eps,min_points=self.min_points))
        n_labels = labels.max() + 1 
        palm     = hands.mean(0)

        l_center = []
        l_index  = []
        l_dist   = []
        l_num    = []
        for label in range(n_labels):
            index       = np.where(labels == label)[0]
            subpoints   = points[index] 
            l_center.append(subpoints.mean(0))  
            l_index.append(index)
            l_dist.append(np.linalg.norm(subpoints-palm, axis=-1).min())    
            l_num.append(len(index))
        
        if np.argmax(l_num) != np.argmin(l_dist):
            return None
        index    = np.argmax(l_num)
        center   = l_center[index]
        index    = l_index[index]
        if len(self) < self.window_size:
            if len(self) > 1:
                self.add_dr(np.linalg.norm(center-self.xyzs[-1]))
            self.add_r(np.linalg.norm(center))
            self.add_xyz(center)
            return None
        else:
            array = np.array(self.drs)
            mu    = array.mean()
            std   = array.std() 
            self.add_dr(np.linalg.norm(center-self.xyzs[-1]))
            self.add_r(np.linalg.norm(center))
            self.add_xyz(center)
            if (self.drs[-1] - mu) > std*self.std_scale:
                return None 
            else:
                return pcd.select_by_index(index)

    def add_r(self, r:float):
        self.rs.append(r)
        if len(self.rs) > self.window_size:
            self.rs.pop(0)
    def add_dr(self, dr:float):
        self.drs.append(dr)
        if len(self.drs) > self.window_size:
            self.drs.pop(0)
    def add_xyz(self, xyz:np.ndarray):
        self.xyzs.append(xyz)
        if len(self.xyzs) > self.window_size:
            self.xyzs.pop(0)
    



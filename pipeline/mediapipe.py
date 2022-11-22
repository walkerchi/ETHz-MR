from ..io.dataset import Frame,DepthFrame,RGBFrame
from scipy.spatial import ConvexHull
import open3d as o3d
import mediapipe as mp
import cv2
import numpy as np

class RGBMediapipePaintBlack:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.hands_detector=mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.hands_detector = self.hands_detector.__enter__()
    def __call__(self, rgb_frame:RGBFrame)->RGBFrame:
        h, w, _ = rgb_frame.rgb.shape
        results = self.hands_detector.process(cv2.flip(rgb_frame.rgb,1))
        lefthand  = []
        righthand = []
        if results.multi_hand_landmarks is not None:
            for landmarks,handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                if handedness.classification[0].label == "Left":
                    for landmark in landmarks.landmark:
                        x = w - min(int(landmark.x * w), w-1)
                        y = min(int(landmark.y * h), h-1) 
                        lefthand.append([x,y])
                elif handedness.classification[0].label == "Right":
                    for landmark in landmarks.landmark:
                        x = w - min(int(landmark.x * w), w-1)
                        y = min(int(landmark.y * h), h-1) 
                        righthand.append([x,y])
        lefthand = np.array(lefthand)
        righthand = np.array(righthand)
        
        hands = []
        if len(lefthand) > 0:
            lefthand = np.concatenate([lefthand,np.array([[0,0],[0,h-1]])], 0)
            hands.append(lefthand)
        if len(righthand) > 0:
            righthand = np.concatenate([righthand,np.array([[w-1,0],[w-1,h-1]])])
            hands.append(righthand)
        for hand in hands:
            hull = ConvexHull(hand)
            vertices = hand[hull.vertices]
            center   = vertices.mean(0)
            vertices = vertices + self.alpha*(vertices - center)
            vertices = vertices.astype(int)
            vertices[:,0] = np.clip(vertices[:,0],0,w-1)
            vertices[:,1] = np.clip(vertices[:,1],0,h-1)
            src = vertices
            dst = np.roll(vertices, 1, 0)
            topleft        = hull.min_bound + self.alpha*(hull.min_bound - center)
            bottomright    = hull.max_bound + self.alpha*(hull.max_bound - center)
            topleft        = topleft.astype(int)
            bottomright    = bottomright.astype(int)
            topleft[0]     = np.clip(topleft[0], 0, w-1)
            topleft[1]     = np.clip(topleft[1], 0, h-1)
            bottomright[0] = np.clip(bottomright[0], 0, w-1)
            bottomright[1] = np.clip(bottomright[1], 0, h-1)
            pp  = np.stack(np.meshgrid(
                    np.arange(topleft[0],bottomright[0],dtype=int),
                    np.arange(topleft[1],bottomright[1],dtype=int)
                    ),-1).reshape([-1,2])
            src, dst, pp = src[None, ...], dst[None, ...], pp[:,None,:]
            mask = (np.cross(dst - src , pp - src) < 0).all(-1)
            pp  = pp[:, 0, :]
            pp  = pp[mask].astype(int)
            rgb_frame.rgb[pp[:,1],pp[:,0]] = 0
        return  rgb_frame
    def __del__(self):
        self.hands_detector.__exit__(None, None, None)


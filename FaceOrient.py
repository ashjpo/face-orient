#encoding:utf-8
import numpy as np
import math
import cv2
class FaceOrient:
    def __init__(self):
        pass
    
    def face_orient_1(self,points,slow_rate=0,point_type="type4",image=None):
        
        """
                判断人脸角度方法一
        """
        (x1,y1)=points[4]
        (x2,y2)=points[12]
        
        if (x1-x2)==0:
            k=(y1-y2)/0.0001
            b=y1-(x1*(y1-y2)/0.001)
        else:
            k=(y1-y2)/(x1-x2)
            b=y1-(x1*(y1-y2)/(x1-x2))
            
        if k==0:
            k_n=-1*(1/0.0001)
        else:
            k_n=-1*(1/k)
            
        #中点
        x_center=(x1+x2)/2
        y_center=(y1+y2)/2
        #横向和纵向差
        x_mul=x2-x1
        y_mul=y2-y1
        #左点
        x_left=x_center-x_mul*slow_rate
        y_left=y_center-y_mul*slow_rate
        #右点
        x_right=x_center+x_mul*slow_rate
        y_right=y_center+y_mul*slow_rate
        
        if image is not None:
            cv2.circle(image, (int(x_center),int(y_center)), 2, (55, 55, 25), 2)
            cv2.circle(image, (int(x_left),int(y_left)), 2, (55, 55, 25), 2)
            cv2.circle(image, (int(x_right),int(y_right)), 2, (55, 55, 25), 2)
        
        center_b_n=y_center-k_n*x_center
        left_b_n=y_left-k_n*x_left
        right_b_n=y_right-k_n*x_right
        
        center_left_num,center_right_num=self.compute_left_right_point_num(points, k_n, center_b_n, point_type)
        l_left_num,l_right_num=self.compute_left_right_point_num(points, k_n, left_b_n, point_type)
        r_left_num,r_right_num=self.compute_left_right_point_num(points, k_n, right_b_n, point_type)
        all_filter_point_num=center_left_num+center_right_num
        mid_num=all_filter_point_num-l_left_num-r_right_num
        return l_left_num,r_right_num,mid_num
        
        
        
        
        
        
    def compute_left_right_point_num(self,points,k_n,b_n,point_type="type4"):
        """
                用于计算在线的左右数量
        """
        left_num=0
        right_num=0
        for i in range(len(points)):
            if point_type=="type1":
                if i>=48 and i<=67:
                    (x,y)=points[i]
                    now_y=k_n*x+b_n
                    if b_n>0:
                        if (now_y)>y:
                            left_num=left_num+1
                        else:
                            right_num=right_num+1
                    else:
                        if (now_y)>y:
                            right_num=right_num+1
                        else:
                            left_num=left_num+1
            elif point_type=="type2":
                if i>=48 and i<=67 and i!=51 and i!=62 and i!=66 and i!=57:
                    (x,y)=points[i]
                    now_y=k_n*x+b_n
                    if b_n>0:
                        if (now_y)>y:
                            left_num=left_num+1
                        else:
                            right_num=right_num+1
                    else:
                        if (now_y)>y:
                            right_num=right_num+1
                        else:
                            left_num=left_num+1
                            
            elif point_type=="type3":
                if (i>=48 and i<=67 and i!=51 and i!=62 and i!=66 and i!=57) or i==31 or i==32 or i==34 or i==35:
                    (x,y)=points[i]
                    now_y=k_n*x+b_n
                    if b_n>0:
                        if (now_y)>y:
                            left_num=left_num+1
                        else:
                            right_num=right_num+1
                    else:
                        if (now_y)>y:
                            right_num=right_num+1
                        else:
                            left_num=left_num+1
                            
            elif point_type=="type4":
                if i==31 or i==35 or i==48 or i==60 or i==64 or i==54 or i==20 or i==23:
                    (x,y)=points[i]
                    now_y=k_n*x+b_n
                    if b_n>0:
                        if (now_y)>y:
                            left_num=left_num+1
                        else:
                            right_num=right_num+1
                    else:
                        if (now_y)>y:
                            right_num=right_num+1
                        else:
                            left_num=left_num+1
                            
        return left_num,right_num
        
        
        
        
        
        
    
    
    
    
    def face_orient_2(self,im,point):
        """
                判断人脸角度方法二:3d
        """
        size = im.shape
        if size[0] > 700:
            h = size[0] / 3
            w = size[1] / 3
            im = cv2.resize( im, (int( w ), int( h )), interpolation=cv2.INTER_CUBIC )
            size = im.shape
        image_points = np.array([
                                    point[30],     # Nose tip
                                    point[8],     # Chin
                                    point[36],     # Left eye left corner
                                    point[45],     # Right eye right corne
                                    point[48],     # Left Mouth corner
                                    point[54]      # Right mouth corner
                                ], dtype="double")
        # 3D model points.
        model_points = np.array([
                                    (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner
                                 
                                ])
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE )
        if success==False:
            return False,0,0,0
        theta = cv2.norm(rotation_vector, cv2.NORM_L2)
        w = math.cos(theta / 2)
        x = math.sin(theta / 2)*rotation_vector[0][0] / theta
        y = math.sin(theta / 2)*rotation_vector[1][0] / theta
        z = math.sin(theta / 2)*rotation_vector[2][0] / theta
        ysqr = y * y
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        pitch = math.atan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        if t2 > 1.0:
            t2 = 1.0
        if t2 < -1.0:
            t2 = -1.0
        yaw = math.asin(t2)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        roll = math.atan2(t3, t4)
        Y = int((pitch/math.pi)*180)
        X = int((yaw/math.pi)*180)
        Z = int((roll/math.pi)*180)
        
        return True, Y, X, Z
    
    def face_orient_3(self,frame,point,th=1.28,faceboxs=None,special_orient_control=False):
        """
                判断人脸角度方法三
        """
        try:
            if faceboxs is None:
                special_orient_control=False
            elif th>100:
                special_orient_control=False
                
            right_eye,left_eye,nose_top,nose_bottom=self.get_orient_2d_landmark(point)
            
            '''cv2.circle(frame, right_eye, 2, (255, 0, 0), 1)
            cv2.circle(frame, left_eye, 2, (255, 0, 0), 1)
            cv2.circle(frame, nose_top, 2, (255, 0, 0), 1)
            cv2.circle(frame, nose_bottom, 2, (255, 0, 0), 1)'''
            r0,e0=left_eye
            r1,e1=right_eye
            
            t0=int((r0+r1)/2)
            p0=int((e0+e1)/2)
            
            x0,y0=nose_top
            x1,y1=nose_bottom
            
            if x0==x1:
                tempx=0.00001
            else:
                tempx=x0-x1
            
            k1=(y0-y1)/tempx
            b1=y0-k1*x0
            
            
            if r0==r1:
                tempr=0.00001
            else:
                tempr=r0-r1
            
            k2=(e0-e1)/tempr
            b2=e0-k2*r0
            
            a1 = np.array([[-k1,1],[-k2,1]])
            a2 = np.array([b1,b2])
            res=np.linalg.solve(a1,a2)
            traget_x=res[0]
            traget_y=res[1]
            
            #cv2.circle(frame, (int(traget_x), int(traget_y)), 2, (50, 20, 95), 2)
            base_dist=math.sqrt((r1-r0)*(r1-r0)+(e1-e0)*(e1-e0)).real
            base_dist=int(base_dist/2)
            target_dist=float(math.sqrt((traget_x-t0)*(traget_x-t0)+(traget_y-p0)*(traget_y-p0)).real)
            
            if target_dist>th*base_dist:
                return False,target_dist/base_dist,True
    
            else:
                return True,target_dist/base_dist,True  
            
        except:
            return False,10000,False  
        
    def get_orient_2d_landmark(self,point):
        #right_eye
        right_top_x=0
        right_top_y=0
        right_bottom_x=0
        right_bottom_y=0
        
        (x,y)=point[43]
        right_top_x=x
        right_top_y=y
        
        (x,y)=point[44]
        right_top_x=(right_top_x+x)/2
        right_top_y=(right_top_y+y)/2
        (x,y)=point[46]
        right_bottom_x=x
        right_bottom_y=y
        
        (x,y)=point[47]
        right_bottom_x=(right_bottom_x+x)/2
        right_bottom_y=(right_bottom_y+y)/2
        
        right_eye_x=int((right_top_x+right_bottom_x)/2)
        right_eye_y=int((right_top_y+right_bottom_y)/2)
        
        right_eye=(right_eye_x,right_eye_y)
        
        #left_eye
        left_top_x=0
        left_top_y=0
        left_bottom_x=0
        left_bottom_y=0
        
        (x,y)=point[37]
        left_top_x=x
        left_top_y=y
        
        (x,y)=point[38]
        left_top_x=(left_top_x+x)/2
        left_top_y=(left_top_y+y)/2
        (x,y)=point[40]
        left_bottom_x=x
        left_bottom_y=y
        
        (x,y)=point[41]
        left_bottom_x=(left_bottom_x+x)/2
        left_bottom_y=(left_bottom_y+y)/2
        
        left_eye_x=int((left_top_x+left_bottom_x)/2)
        left_eye_y=int((left_top_y+left_bottom_y)/2)
        
        left_eye=(left_eye_x,left_eye_y)
        
        nose_top=point[30]
        nose_bottom=point[33]
        
        return right_eye,left_eye,nose_top,nose_bottom
                    
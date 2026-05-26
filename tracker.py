import numpy as np

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        
    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2         #Center x
            cy = (y1 + y2) // 2         #Center y
            same_object_detected = False
            
            for obj_id, pt in self.center_points.items():           #Loop through stored objects (format of stored objects - { 0:(120,70), 1:(400,250),})
                dist = np.linalg.norm(np.array([cx, cy]) - np.array(pt))        #Calculates Euclidean Distance
                if dist < 35:       #Treat as same object
                    self.center_points[obj_id] = (cx,cy,)       #Move stored center to latest position - update center
                    objects_bbs_ids.append([x1, y1, x2, y2, obj_id])        #Save box + ID
                    same_object_detected = True         #Mark as found
                    break
                
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy,)           #Store new center
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count,])            #Add result
                self.id_count += 1          #Next ID ready
                
        return objects_bbs_ids          #Return tracked objects
                
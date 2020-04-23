from __future__ import division

class Model():
    def __init__(self):       
        # print("pre-trained model loading ...")        
        print("model loaded successfully")       
   

    def predict(self, frame):
        pred_result = []
        points = [0, 0, 0, 0]
        label = "car"
        score = 1.0
        shape = {"points": points, "label": label, "score": score}  
        pred_result.append(shape)
        return frame, pred_result        


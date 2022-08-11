import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("test.csv")
thresholds=list(np.array(list(range(0,105,5)))/100)
roc_point=[]

for threshold in thresholds:
    tp = 0; fp = 0; fn = 0; tn = 0
    Precision=0;Specificity=0;Accuracy=0;Recall=0
    for index, instance in df.iterrows():
        actual = instance["actual"]
        prediction = instance["prediction"]
            
        if prediction >= threshold:
            prediction_class = 1
        else:
            prediction_class = 0
        if prediction_class == 1 and actual == 1:
            tp = tp + 1 
        elif actual == 1 and prediction_class == 0:
            fn = fn + 1 
        elif actual == 0 and prediction_class == 1: 
            fp = fp + 1
        elif actual == 0 and prediction_class == 0:
            tn = tn + 1

    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    roc_point.append([tpr,fpr])
    if tp==fp==0:
        pass
    else:
        Precision=tp/(tp+fp)
    if tn==fn==0:
        pass
    else:    
        Specificity=tn/(tn+fn)
    Accuracy=(tp+tn)/(tp+tn+fp+fn)
    False_positive_rate=fp/(tn+fp)
    Recall=tp/(tp+fn)
    print(Precision,Specificity,Accuracy,Recall)
pivot=pd.DataFrame(roc_point,columns=["x","y"])
pivot["thresholds"]=thresholds
plt.scatter(pivot.y,pivot.x)
plt.plot([0,1])
plt.show()
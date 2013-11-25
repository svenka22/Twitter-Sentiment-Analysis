from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class Metrics:
    def calculateClassifierMetrics(self, actualClassList, predictedClassList):
        #actualClassList=List[]; predictedClassList=List[]
        #convert label values to int values
        label_true = self.convertLabel(actualClassList)
        label_false = self.convertLabel(predictedClassList)
        Accuracy = accuracy_score(label_true, label_false)
        recall,precision,fbeta_score,support = precision_recall_fscore_support(label_true, label_false, average='macro')
        #Display the metrics
        print "Accuracy using scikit: ",Accuracy
        print "Precision: ",precision
        print "Recall: ",recall
        print "F-Score: ",fbeta_score
        print "Support: ",support
    
    # convert the text label to int label for scikit metrics package
    def convertLabel(self, labelList):
        output = list()
        for lbl in labelList:
            if lbl == "Cessation":
                output.append(1)
            elif lbl == "No Cessation":
                output.append(0)
        return output


                
        
        
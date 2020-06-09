# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:41:13 2019

@author: Adithya Kommini
"""

def modelSave(modelName,no,report):
    name = "trained_model_" + str(no) +".h5"
    reportname = "savedModels\\report_model_" + str(no) +".txt"
    modelName.save("savedModels\\" + name)
    sample = open(reportname, 'w') 
    print(report,file = sample)
    sample.close() 

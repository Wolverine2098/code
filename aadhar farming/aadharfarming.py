# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:26:08 2019

@author: Logan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv

print("Welcome to Aadhar farming fertilizer recommendation system")

farmer_val=pd.read_csv('farmer_master.csv')
crop_area=farmer_val.iloc[0:1,4:5].values
farmer_name=farmer_val.iloc[0:1,2:3].values
aadhar_no= int(farmer_val.iloc[0:1,1:2].values)
print("Farmer name:",farmer_name)
print("Farmer aadhar number:",aadhar_no)
Crop_id=pd.read_csv('crop_master.csv')
crop_idval=Crop_id.iloc[0:1,0:1].values
                       
if crop_idval==1:
    print("Crop name : Paddy")
    dataset=pd.read_csv('Paddy.csv')
    X=dataset.iloc[:,1:2].values

    X=dataset.iloc[:,1:2].values  
    Y=dataset.iloc[:,2:3].values              
    Y1=dataset.iloc[:,3:4].values
    Y2=dataset.iloc[:,4:5].values
    Y3=dataset.iloc[:,5:6].values
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/4,random_state=0)    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit_transform(X_train,Y_train)
    # Fittting regression using SVR to the training set
    from sklearn.svm import SVR
    regressor=SVR(kernel='linear',C=1,gamma='auto',epsilon=1000)
    regressor.fit(X_train,Y_train.ravel()) 
    Y_pres=regressor.predict(X_test)
    y_predict=regressor.predict(crop_area*1000)
    slope=y_predict/(crop_area*1000)
    out=float((slope*crop_area)/4.0)
    print("Amount of DAP required per month for 4 months is",out,"Kg")
    
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y1,test_size=1/4,random_state=0)    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit_transform(X_train,Y_train)
    # Fittting regression using SVR to the training set
    from sklearn.svm import SVR
    regressor=SVR(kernel='linear',C=1,gamma='auto',epsilon=1)
    regressor.fit(X_train,Y_train.ravel()) 
    Y_pres=regressor.predict(X_test)
    y_predict=regressor.predict(crop_area*1000)
    slope=y_predict/(crop_area*1000)
    out1=float((slope*crop_area)/4.0)
    print("Amount of Urea required per month for 4 months is",out1,"Kg")
    
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y2,test_size=1/4,random_state=0)    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit_transform(X_train,Y_train)
    # Fittting regression using SVR to the training set
    from sklearn.svm import SVR
    regressor=SVR(kernel='linear',C=1,gamma='auto',epsilon=27000)
    regressor.fit(X_train,Y_train.ravel()) 
    Y_pres=regressor.predict(X_test)
    y_predict=regressor.predict(crop_area*1000)
    slope=y_predict/(crop_area*1000)
    out2=float((slope*crop_area)/4.0)
    print("Amount of Super Phospate required per month for 4 months is",out2,"Kg")
    
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y3,test_size=1/4,random_state=0)    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit_transform(X_train,Y_train)
    # Fittting regression using SVR to the training set
    from sklearn.svm import SVR
    regressor=SVR(kernel='linear',C=1,gamma='auto',epsilon=2290)
    regressor.fit(X_train,Y_train.ravel()) 
    Y_pres=regressor.predict(X_test)
    y_predict=regressor.predict(crop_area*1000)
    slope=y_predict/(crop_area*1000)
    out3=float((slope*crop_area)/4.0)
    print("Amount of Ammonium Sulphate required per month for 4 months is",out3,"Kg")
    
    
else:
    print("Crop name: Bajra")
    dataset=pd.read_csv('Bajra.csv')
    X=dataset.iloc[:,1:2].values  
    Y=dataset.iloc[:,2:3].values              
    Y1=dataset.iloc[:,3:4].values
    Y2=dataset.iloc[:,4:5].values
    Y3=dataset.iloc[:,5:6].values
        
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=1/4,random_state=0)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit_transform(x_train,y_train)

    from sklearn.tree import DecisionTreeRegressor
    cid = DecisionTreeRegressor(max_depth=9)
    cid=cid.fit(x_train,y_train)
    y_predict=cid.predict(crop_area*1000)
    slope=y_predict/(crop_area*1000)
    out=float((slope*crop_area)/4.0)
    print("Amount of DAP required per month for 4 months is",out,"Kg")
        
        
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(X,Y1,test_size=1/4,random_state=0)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit_transform(x_train,y_train)

    from sklearn.tree import DecisionTreeRegressor
    cid = DecisionTreeRegressor(max_depth=9)
    cid=cid.fit(x_train,y_train)
    y_predict=cid.predict(crop_area*1000)
    slope=y_predict/(crop_area*1000)
    out1=float((slope*crop_area)/4.0)
    print("Amount of Urea required per month for 4 months is",out1,"Kg")
        
        
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(X,Y2,test_size=1/4,random_state=0)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit_transform(x_train,y_train)

    from sklearn.tree import DecisionTreeRegressor
    cid = DecisionTreeRegressor(max_depth=9)
    cid=cid.fit(x_train,y_train)
    y_predict=cid.predict(crop_area*1000)
    slope=y_predict/(crop_area*1000)
    out2=float((slope*crop_area)/4.0)
    print("Amount of Super Phosphate required per month for 4 months is",out2,"Kg")
        
        
        
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(X,Y3,test_size=1/4,random_state=0)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit_transform(x_train,y_train)
 
    from sklearn.tree import DecisionTreeRegressor
    cid = DecisionTreeRegressor(max_depth=9)
    cid=cid.fit(x_train,y_train)
    y_predict=cid.predict(crop_area*1000)
    slope=y_predict/(crop_area*1000)
    out3=float((slope*crop_area)/4.0)
    print("Amount of Ammonium Sulphate required per month for 4 months is",out3,"Kg")
        
with open('output.csv','w',newline='') as f:
    write=csv.writer(f) 
    write.writerow(['DAP','UREA','Super phosphate','Ammonium sulphate'])
    write.writerow([out,out1,out2,out3])       
        

    
    
##Importing all the needed pacakges 
import os

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

##add code to input file from user
data=pd.read_csv("C:\data\heart_disease_data.csv") ##loading in the data set

temp = ""
shape=data.shape


def columnnames():#function for columnnames
    headers = list(data.columns.values)
    print("The names of the variables are : ", headers)




def picktarget(temp):#function to pick a target value (dependent variable)
    columnnames()
    if 'target' in data.columns:
        choice = input("A column with the name 'target' exists. Would you like to rename it to pick a new target? (Y/N): ")
        if choice == "Y":
            rename = input("Provide a new name for the column 'target': ")
            data.rename(columns={'target': rename}, inplace=True)
            column = input("Pick a column to make target: ")
            if column in data.columns:
                temp = column
                data.rename(columns={column: 'target'}, inplace=True)
            else:
                print("This column is not a valid column name")
    else:
        column = input("Pick a column to make target: ")
        if column in data.columns:
            temp = column
            data.rename(columns={column: 'target'}, inplace=True)
        else:
            print("This column is not a valid column name")





def pred(model, scaler, data_columns):#function to predict based on selected model
    while (1):
        typepred = int(input("Do you want to predict 1 output or multiple or exit prediction ? (1=1, 2=many, 3=end): "))
        
        if typepred == 1:
            dummyval = []
            for col in data_columns:
                x = float(input(f"Enter the value for {col}: "))  
                dummyval.append(x)
            dummyval = scaler.transform([dummyval])  # Scale the input
            prediction = model.predict(dummyval)
            print(f"Predicted value of target: {prediction}")
        
        elif typepred == 2:
            num_predictions = int(input("Enter the number of predictions: "))
            predictions = []
            for j in range(num_predictions):
                dummyval = []
                print(f"Entering values for prediction {j+1}:")
                for col in data_columns:
                    x = float(input(f"Enter the value for {col}: "))
                    dummyval.append(x)
                predictions.append(dummyval)
            predictions = scaler.transform(predictions)  # Scale the inputs
            predictions = model.predict(predictions)
            for i, pred in enumerate(predictions):
                print(f"Predicted value of target {i+1}: {pred}")

        else:
            break


#the following are functions for each type of regression model to make predictions on

def logisticreg(x,y,sol): # find best solver out of "newton-cg","lbfgs","liblinear","sag","saga" 
    smote = SMOTE(random_state=0)
    x_os, y_os = smote.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x_os, y_os, test_size=0.3, random_state=0)

    scaler = StandardScaler().fit(x_train)
    x_train_sc = scaler.transform(x_train)

    model = LogisticRegression(solver=sol)
    model.fit(x_train_sc, y_train)

    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(LogisticRegression(solver=sol), x_train_sc, y_train, cv=kfold)
    
    return results.mean() * 100, model, scaler





def lassoreg(x,y,i):#alpha 10^-4-10^4 with 100 data values#Lasso(alpha=i)
    smote = SMOTE(random_state=0)
    x_os, y_os = smote.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x_os, y_os, test_size=0.3, random_state=0)

    scaler = StandardScaler().fit(x_train)
    x_train_sc = scaler.transform(x_train)

    model = Lasso(alpha=i)
    model.fit(x_train_sc, y_train)

    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(Lasso(alpha=i), x_train_sc, y_train, cv=kfold)
    
    return results.mean() * 100, model, scaler




def ridgereg(x,y,i):#alpha 10^-4-10^4 with 100 data values#Ridge(alpha=i)
    smote = SMOTE(random_state=0)
    x_os, y_os = smote.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x_os, y_os, test_size=0.3, random_state=0)

    scaler = StandardScaler().fit(x_train)
    x_train_sc = scaler.transform(x_train)

    model = Ridge(alpha=i)
    model.fit(x_train_sc, y_train)

    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(Ridge(alpha=i), x_train_sc, y_train, cv=kfold)
    
    return results.mean() * 100, model, scaler




def rforestreg(x,y,i):#n_estimator 0-500 inc 50 find best#RandomForestRegressor(n_estimators=i)
    smote = SMOTE(random_state=0)
    x_os, y_os = smote.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x_os, y_os, test_size=0.3, random_state=0)

    scaler = StandardScaler().fit(x_train)
    x_train_sc = scaler.transform(x_train)

    model = RandomForestRegressor(n_estimators=i)
    model.fit(x_train_sc, y_train)

    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(RandomForestRegressor(n_estimators=i), x_train_sc, y_train, cv=kfold)
    
    return results.mean() * 100, model, scaler




def svreg(x,y,ker):#kernel="linear","polynomial","rbf","sigmoid"#SVR(kernel=ker)
    smote = SMOTE(random_state=0)
    x_os, y_os = smote.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x_os, y_os, test_size=0.3, random_state=0)

    scaler = StandardScaler().fit(x_train)
    x_train_sc = scaler.transform(x_train)

    model = SVR(kernel=ker)
    model.fit(x_train_sc, y_train)

    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(SVR(kernel=ker), x_train_sc, y_train, cv=kfold)
    
    return results.mean() * 100, model, scaler




def MLPreg(x,y,act,sol):#activation can include :['identity', 'logistic', 'tanh', 'relu'] and solver can have : ['lbfgs', 'sgd', 'adam']
    #MLPRegressor(hidden_layer_sizes=(100,), activation=act, solver=sol, max_iter=1000, random_state=42)
    smote = SMOTE(random_state=0)
    x_os, y_os = smote.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x_os, y_os, test_size=0.3, random_state=0)

    scaler = StandardScaler().fit(x_train)
    x_train_sc = scaler.transform(x_train)

    model = MLPRegressor(hidden_layer_sizes=(100,), activation=act, solver=sol, max_iter=2500, random_state=42)
    model.fit(x_train_sc, y_train)

    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(MLPRegressor(hidden_layer_sizes=(100,), activation=act, solver=sol, max_iter=1000, random_state=42), x_train_sc, y_train, cv=kfold)
    
    return results.mean() * 100, model, scaler




print("Here is a Description Table of your data : \n\n")# data description is present here 

print(data.describe())




print("One sec while we drop duplicates and null values....\nThis will help you build accuracy...\n")
data.drop_duplicates(inplace=True)
print("You now have\n",data.isnull().sum(),"\nNull values &",data.duplicated().sum(),"Duplicate values\n\n")

print("Please pick your Target varible of the following (Target is what you are measuring for)")
temp = picktarget( temp)


while(1):

    print("Proceeding.....\n\n Choose what you would like to do of the following options :-")#inital options
    type=int(input("1.) Heatmap\n2.) Bar Plot\n3.) Pie Chart\n4.) Box N' Whiskers\n5.) Scatter/Regression Plot\n6.) Line Plot\n7.) Cumulative Graph --- (Against Target specifically)\n8.) Predictive models\n9.) Repick the Target Variable \n10.)Done\n\nEnter your Choice : "))
    
    
    
    
    if(type==1):#heatmap code
        plt.figure(figsize=(20,8))
        sns.heatmap(data.corr(),annot=True)
        plt.title("Correlation Heatmap")
        print(plt.show())
        data_x=data
        c=['black','purple','orange','grey']
        data_x.corrwith(data['target']).plot.bar(figsize=(20,10),fontsize=15,title='Target Correlation',rot=45,grid=True,color=c)
        print(plt.show())
        continue




    elif(type==2):#bar plot code
        while(1):
            ver=input("Enter the type of comparision against Target or between variables:-(Comp or Target) : ")
            columnnames()
            if(ver=="Comp"):
                column = input("Enter the first parameter for comparision : ")
                column1 = input("Enter the second parameter for comparision : ")
            
                if column in data.columns:
                    if column1 in data.columns:
                        data.groupby(f'{column}')[f'{column1}'].mean().plot(kind='bar')
                        plt.title(f"Bar Plot of {column} against {column1}")
                        print(plt.show())
                else:
                    print(f"Column/Columns not found in data.")
            elif ver=="Target":
                column = input("Enter the parameter for comparision : ")
                if column in data.columns:
                        data.groupby(f'{column}')['target'].mean().plot(kind='bar')
                        plt.title(f"Bar Plot of {column} against Target")
                        print(plt.show())
                else:
                    print(f"Column not found in data.")
            opt=input("Do you want to make another Bar plot? (Y/N) : ")
            if opt=="N":
                break
        continue




    elif(type==3):#pie chart code
        while(1):
            columnnames()
            column = input("Choose which parameter you would like to make a pie chart for : ")
            if column in data.columns:
                plt.figure(figsize=(7,4))
                data[f'{column}'].value_counts().plot(kind='pie',autopct='%1.1f%%')
                print(plt.show())
            else:
                print(f"The column {column} is not present in the data columns")
            opt=input("Do you want to make another Pie chart? (Y/N) : ")
            if opt=="N":
                break
        continue




    elif(type==4):#box plot code 
        while(1):
            columnnames()
            column = input("Enter the parameter for comparision : ")
            if column in data.columns:
                    plt.figure(figsize=(30,20))
                    sns.boxplot(x=f"{column}", data=data,hue='target')
                    plt.title(f"Box Plot of {column} against target")
                    print(plt.show())
            else:
                print(f"Column not found in data.")
            opt=input("Do you want to make another Box plot? (Y/N) : ")
            if opt=="N":
                break
        continue




    elif(type==5):#scatter plot code
        while(1):
            columnnames()
            column = input("Enter the first parameter for comparision (x-axis): ")
            column1 = input("Enter the second parameter for comparision (y-axis): ")
            
            if column in data.columns:
                if column1 in data.columns:
                    sns.scatterplot(data=data, x=f'{column}',y=f'{column1}', hue='target')
                    plt.title(f"Scatter Plot of {column} against {column1}")
                    print(plt.show())
                    var=input("Do you want a regression of this plot ? (Y/N) : ")
                    if(var=="Y"):
                        sns.regplot(data=data, x=f'{column}',y=f'{column1}')
                        plt.title(f"Regression Plot of {column} against {column1}")
                        print(plt.show())
                else:
                    print(f"Column/Columns not found in data.")
            opt=input("Do you want to make another Scatter/Regression plot? (Y/N) : ")
            if opt=="N":
                break
        continue




    elif(type==6):# Line plot
        while(1):
            columnnames()
            column = input("Enter the parameter for comparision : ")
            if column in data.columns:
                    plt.plot(data[f'{column}'], data['target'])
                    plt.title(f'Line plot of {column} against target')
                    print(plt.show())
            else:
                print(f"Column not found in data.")
            opt=input("Do you want to make another Line plot? (Y/N) : ")
            if opt=="N":
                break
        continue




    elif (type==7):#cummilative plots (not included in open source version)
        while True:
                print("REMINDER****(All Cumulative graphs are made against the target variable)****\n\n")
                graph = int(input("1.) Bar Plot\n2.) Pie Chart\n3.) Box N' Whiskers\n4.) Scatter/Regression Plot\n5.) Line Plot\n6.) Back to Main Menu\n\nEnter your Choice : "))
                
                if graph == 1:
                    plt.figure(figsize=(30, 20))
                    for idx, col in enumerate(data.columns):
                        plt.subplot(3, 5, idx + 1)
                        ax = sns.countplot(x=col, hue='target', data=data)
                        ax.set_title(f'Bar Plot of {col}')
                        ax.set_xlabel(col)
                        ax.set_ylabel('Count')
                    plt.show()
                elif graph == 2:
                    plt.figure(figsize=(30, 20))
                    for idx, col in enumerate(data.columns):
                        plt.subplot(3, 5, idx + 1)
                        data[col].value_counts().plot(kind='pie', autopct='%1.1f%%')
                        plt.title(f'Pie Chart of {col}')
                    plt.show()
                elif graph == 3:
                    plt.figure(figsize=(30, 20))
                    for idx, col in enumerate(data.columns.drop(['sex', 'exang', 'target', 'fbs', 'slope'])):
                        plt.subplot(6, 4, idx + 1)
                        ax = sns.boxplot(x='target', y=col, data=data)
                        ax.set_title(f'Box Plot of {col}')
                        ax.set_xlabel('Target')
                        ax.set_ylabel(col)
                    plt.show()
                elif graph == 4:
                    var1=int(input("Which type of scatter plot do you want?\n\n1.) Data Uniquness\n2.) Correlation Level (20%-30% is recommended for highly volatile values)\n\nEnter the type : "))
                    if var1==1:
                        while(1):
                            lvl=int(input("Enter the number of minimum unique values to have a wider range of scatter (Pick a value above 5 for more scatter) : "))
                            if lvl<5:
                                print("Please pick a value greater than 5")
                            else:
                                break
                        plt.figure(figsize=(30, 20))
                        shape = data.shape
                        subplot_idx = 1
                        for i in range(shape[1]):
                            for j in range(i + 1, shape[1]):
                                if data[data.columns[i]].nunique() > lvl and data[data.columns[j]].nunique() > lvl:
                                    plt.subplot(3, 5, subplot_idx)
                                    ax = sns.scatterplot(data=data, x=data.columns[i], y=data.columns[j], hue='target')
                                    ax.set_title(f'Scatter Plot of {data.columns[i]} vs {data.columns[j]}')
                                    ax.set_xlabel(data.columns[i])
                                    ax.set_ylabel(data.columns[j])
                                    subplot_idx += 1
                                    if subplot_idx > 15:
                                        break
                            if subplot_idx > 15:
                                break
                        plt.show()
                        choice = input("Do you want a cumulative regression plot? (Y/N) : ")
                        if choice.upper() == "Y":
                            plt.figure(figsize=(30, 20))
                            subplot_idx = 1
                            for i in range(shape[1]):
                                for j in range(i + 1, shape[1]):
                                    if data[data.columns[i]].nunique() > lvl and data[data.columns[j]].nunique() > lvl:
                                        plt.subplot(3, 5, subplot_idx)
                                        ax = sns.regplot(data=data, x=data.columns[i], y=data.columns[j])
                                        ax.set_title(f'Regression Plot of {data.columns[i]} vs {data.columns[j]}')
                                        ax.set_xlabel(data.columns[i])
                                        ax.set_ylabel(data.columns[j])
                                        subplot_idx += 1
                                        if subplot_idx > 15:
                                            break
                                if subplot_idx > 15:
                                    break
                            plt.show()
                    elif var1==2:
                        lvl=float(input("Enter the percent correlation between variables (ex. 20 = 20% corr between variables) (Spearman Correlation is used) : "))
                        lvl=lvl/100
                        nlvl=-lvl
                        plt.figure(figsize=(30, 20))
                        shape = data.shape
                        subplot_idx = 1
                        for i in range(shape[1]):
                            for j in range(i + 1, shape[1]):
                                if (data[data.columns[i]].corr(data[data.columns[j]], method='spearman') >= lvl or data[data.columns[i]].corr(data[data.columns[j]], method='spearman') <= nlvl):
                                    plt.subplot(3, 5, subplot_idx)
                                    ax = sns.scatterplot(data=data, x=data.columns[i], y=data.columns[j], hue='target')
                                    ax.set_title(f'Scatter Plot of {data.columns[i]} vs {data.columns[j]}')
                                    ax.set_xlabel(data.columns[i])
                                    ax.set_ylabel(data.columns[j])
                                    subplot_idx += 1
                                    if subplot_idx > 15:
                                        break
                            if subplot_idx > 15:
                                break
                        plt.show()
                        choice = input("Do you want a cumulative regression plot? (Y/N) : ")
                        if choice.upper() == "Y":
                            plt.figure(figsize=(30, 20))
                            subplot_idx = 1
                            for i in range(shape[1]):
                                for j in range(i + 1, shape[1]):
                                    if (data[data.columns[i]].corr(data[data.columns[j]], method='spearman') >= lvl or data[data.columns[i]].corr(data[data.columns[j]], method='spearman') <= nlvl):
                                        plt.subplot(3, 5, subplot_idx)
                                        ax = sns.regplot(data=data, x=data.columns[i], y=data.columns[j])
                                        ax.set_title(f'Regression Plot of {data.columns[i]} vs {data.columns[j]}')
                                        ax.set_xlabel(data.columns[i])
                                        ax.set_ylabel(data.columns[j])
                                        subplot_idx += 1
                                        if subplot_idx > 15:
                                            break
                                if subplot_idx > 15:
                                    break
                            plt.show()     
                elif graph == 5:
                    plt.figure(figsize=(30, 20))
                    shape = data.shape
                    subplot_idx = 1
                    for i in range(shape[1]):
                        for j in range(i + 1, shape[1]):
                            plt.subplot(3, 5, subplot_idx)
                            plt.plot(data[data.columns[i]], data[data.columns[j]])
                            plt.title(f'Line Plot of {data.columns[i]} vs {data.columns[j]}')
                            plt.xlabel(data.columns[i])
                            plt.ylabel(data.columns[j])
                            subplot_idx += 1
                            if subplot_idx > 15:
                                break
                        if subplot_idx > 15:
                            break
                    plt.show()
                elif graph == 6:
                    break
                else:
                    print("Invalid Input")

    elif (type==8):#prediction codes and selection of best models
        opt1 = input("Do you want to proceed analysis with the current column acting as the target (Dependent variable we are testing for) (Y/N) : ")
        if opt1 == "N":
            temp = picktarget(temp)
            
        x = data.drop(['target'], axis=1).values
        y = data['target'].values

        # finding the best of the best 
        overall = [0] * 6

        # Logistic Regression
        logisticsol = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
        logmax = 0
        logsolver = ""
        for sol in logisticsol:
            temp,_,_= logisticreg(x, y, sol)
            if temp > logmax:
                logmax = temp
                logsolver = sol
        overall[0] = logmax


        # Lasso Regression
        lassomax = 0
        lassoalpha = 0
        val = np.logspace(-4, 4, 100)
        for alpha in val:
            temp,_,_= lassoreg(x, y, alpha)
            if temp > lassomax:
                lassomax = temp
                lassoalpha = alpha
        overall[1] = lassomax

        # Ridge Regression
        ridgemax = 0
        ridgealpha = 0
        val = np.logspace(-4, 4, 100)
        for alpha in val:
            temp,_,_= ridgereg(x, y, alpha)
            if temp > ridgemax:
                ridgemax = temp
                ridgealpha = alpha
        overall[2] = ridgemax


        # Random Forest Regression 
        randommax = 0
        randomest = 0
        for est in range(1, 501, 50):
            temp,_,_= rforestreg(x, y, est)
            if temp > randommax:
                randommax = temp
                randomest = est
        overall[3] = randommax

        # SVRegression
        svrkernel = ["linear", "poly", "rbf", "sigmoid"]
        svrmax = 0
        svrker = ""
        for kernel in svrkernel:
            temp,_,_= svreg(x, y, kernel)
            if temp > svrmax:
                svrmax = temp
                svrker = kernel  
        overall[4] = svrmax 

        # mlp regression
        mlpact = ['identity', 'logistic', 'tanh', 'relu']
        mlpsol = ['sgd', 'adam']
        mlpmax = 0
        mlpactivation = ""
        mlpsolver = ""
        for act in mlpact:
            for sol in mlpsol:
                temp,_,_= MLPreg(x, y, act, sol)
                if temp > mlpmax:
                    mlpmax = temp
                    mlpactivation = act
                    mlpsolver = sol
        overall[5] = mlpmax

        best = max(overall)
        index = overall.index(best)

        # Mapping index to model names
        models = [
            "Logistic Regression",
            "Lasso Regression",
            "Ridge Regression",
            "Elastic Regression",
            "Random Forest Regression",
            "SV Regression",
            "MLP Regression"
        ]
        while (1):
            print(f"The most accurate model for this data set is {models[index]}")
            if index == 0:
                print(f"Solver Type: {logsolver}\nAccuracy percent: {best}%")
            elif index == 1:
                print(f"Alpha Value: {lassoalpha}\nAccuracy percent: {best}%")
            elif index == 2:
                print(f"Alpha Value: {ridgealpha}\nAccuracy percent: {best}%")
            elif index == 3:
                print(f"N_Estimator Value: {randomest}\nAccuracy percent: {best}%")
            elif index == 4:
                print(f"Kernel Type: {svrker}\nAccuracy percent: {best}%")
            elif index == 5:
                print(f"Activation Type: {mlpactivation}\nSolver Type: {mlpsolver}\nAccuracy percent: {best}%")

            popt = int(input("Would you like to proceed with this regression? (Y=1, N=0, Exit=2): "))

            if popt == 0:
                rtype = int(input("Enter the type of Regression to be used\n\n1.)Logistic Regression\n2.)Lasso Regression\n3.)Ridge Regression\n4.)Random Forest Regression\n5.)SV Regression\n6.)MLP Regression\n Enter the type of regression you would like to conduct :"))

                if rtype == 1:
                    calc = int(input(f"The most accurate setting is Solver Type: {logsolver}\nAccuracy percent: {best}%\nWould you like to proceed with this? (Y=1/N=0): "))
                    if calc == 1:
                        b, model, scaler = logisticreg(x, y, logsolver)
                        print(f"Your accuracy with this model is: {b}")
                        pred(model, scaler, data.columns.drop('target'))
                    else:
                        a = int(input(f"Pick a solver out of the following 1-5 {logisticsol} : "))
                        b, model,scaler = logisticreg(x, y, logisticsol[a - 1])
                        print(f"Your accuracy with this model of solver - {logisticsol[a - 1]} is:", b)
                        pred(model, scaler, data.columns.drop('target'))

                elif rtype == 2:
                    calc = int(input(f"The most accurate setting is Alpha Of: {lassoalpha}\nAccuracy percent: {best}%\nWould you like to proceed with this? (Y=1/N=0): "))
                    if calc == 1:
                        b, model,scaler = lassoreg(x, y, lassoalpha)
                        print("Your accuracy with this model is:", b)
                        pred(model, scaler, data.columns.drop('target'))
                    else:
                        a = float(input("Pick an alpha between 0.0001 and 10000: "))
                        b, model,scaler = lassoreg(x, y, a)
                        print(f"Your accuracy with this model of alpha - {a} is:", b)
                        pred(model, scaler, data.columns.drop('target'))

                elif rtype == 3:
                    calc = int(input(f"The most accurate setting is Alpha Of: {ridgealpha}\nAccuracy percent: {best}%\nWould you like to proceed with this? (Y=1/N=0): "))
                    if calc == 1:
                        b, model,scaler = ridgereg(x, y, ridgealpha)
                        print("Your accuracy with this model is:", b)
                        pred(model, scaler, data.columns.drop('target'))
                    else:
                        a = float(input("Pick an alpha between 0.0001 and 10000: "))
                        b, model,scaler = ridgereg(x, y, a)
                        print(f"Your accuracy with this model of alpha - {a} is:", b)
                        pred(model, scaler, data.columns.drop('target'))

                elif rtype == 4:
                    calc = int(input(f"The most accurate setting is N_estimator of: {randomest}\nAccuracy percent: {best}%\nWould you like to proceed with this? (Y=1/N=0): "))
                    if calc == 1:
                        b, model,scaler = rforestreg(x, y, randomest)
                        print("Your accuracy with this model is:", b)
                        pred(model, scaler, data.columns.drop('target'))
                    else:
                        a = int(input("Pick an Estimator between 1 and 500: "))
                        b, model,scaler = rforestreg(x, y, a)
                        print(f"Your accuracy with this model of estimator - {a} is:", b)
                        pred(model, scaler, data.columns.drop('target'))

                elif rtype == 5:
                    calc = int(input(f"The most accurate setting is Kernel of: {svrker}\nAccuracy percent: {best}%\nWould you like to proceed with this? (Y=1/N=0): "))
                    if calc == 1:
                        b, model,scaler = svreg(x, y, svrker)
                        print("Your accuracy with this model is:", b)
                        pred(model, scaler, data.columns.drop('target'))
                    else:
                        a = int(input("Pick a Kernel (1-4) out of: "))
                        b, model,scaler = svreg(x, y, svrkernel[a - 1])
                        print(f"Your accuracy with this model of kernel - {svrkernel[a - 1]} is:", b)
                        pred(model, scaler, data.columns.drop('target'))

                elif rtype == 6:
                    calc = int(input(f"The most accurate setting is Activation type of: {mlpactivation} And solver of: {mlpsolver}\nAccuracy percent: {best}%\nWould you like to proceed with this? (Y=1/N=0): "))
                    if calc == 1:
                        b, model,scaler = MLPreg(x, y, mlpactivation, mlpsolver)
                        print("Your accuracy with this model is:", b)
                        pred(model, scaler, data.columns.drop('target'))
                    else:
                        a = int(input("Pick an Activator (1-4) out of: "))
                        c = int(input("Pick a Solver (1-3) out of: "))
                        b, model,scaler = MLPreg(x, y, mlpact[a - 1], mlpsol[c - 1])
                        print(f"Your accuracy with this model of activator - {mlpact[a - 1]} and solver - {mlpsol[c - 1]} is:", b)
                        pred(model, scaler, data.columns.drop('target'))

            elif popt == 1:
                if index == 0:
                    _,model,scaler = logisticreg(x, y, logsolver)
                    pred(model, scaler, data.columns.drop('target'))
                elif index == 1:
                    _,model,scaler = lassoreg(x, y, lassoalpha)
                    pred(model, scaler, data.columns.drop('target'))
                elif index == 2:
                    _,model,scaler = ridgereg(x, y, ridgealpha)
                    pred(model, scaler, data.columns.drop('target'))
                elif index == 3:
                    _,model,scaler = rforestreg(x, y, randomest)
                    pred(model, scaler, data.columns.drop('target'))
                elif index == 4:
                    _,model,scaler = svreg(x, y, svrker)
                    pred(model, scaler, data.columns.drop('target'))
                elif index == 5:
                    _,model,scaler = MLPreg(x, y, mlpactivation, mlpsolver)
                    pred(model, scaler, data.columns.drop('target'))
            
            
            elif popt == 2:
                break

    elif(type==9):
        temp = picktarget( temp)


    elif(type==10):
        break

    else:
        print("Invalid Input")
        continue


print(" Visit us again soon !")

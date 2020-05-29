#README 0) On default parameters, this script will first train and test a random forest, then print statistics of the test before exporting all predicted results to a file called PREDICTED_GRADES.csv"
#       1) In order to run this script, make sure the file paths are correct within the import CSV section ( marked by a #***!!!***# for ease of access)
#               Note: The .cvs files must be on the same file level as this script.
#       2) In order to select model and type of classifier, edit the parameters below.
#       3) If you wish to use this model to predict your own grades from your own data, see the section on this immediately after the #IMPORT MODULES SECTION

#USER PARAMETERS SECTION (EDIT THIS SECTION TO SELECT WHAT YOU WANT TO DO)

#set model to "forest" to run random forest training and statistics, or "svc" to run svc training and statistics
model = "forest"

#set binary to:
#   0 to evaluate "Withdrawn" and "Distinction" as seperate labels,
#   1 to count them as being "Fail"s and "Pass"es respectively, making the model a binary classifier, or
#   2 to remove "Withdrawn"s and "Distinction"s entirely.
#       Note: any number other than 0,1 or 2 will be treated as a 0.
binary = 0

# If you wish to export a file with a list of all students' modules' predicted grades, set export_predicted to 1, if not, set to 0
#   If you select this option your grades will be exported to a file called "PREDICTED_GRADES.csv" after statistics have been printed.
#       Note: any number other than 0 or 1 will be treated as a 0.
export_predicted = 1


#IMPORT MODULES SECTION
print("Importing modules...")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import sys
import datetime

#***!!!***#
#SECTION TO IMPORT CSV FILES FROM THE OULAD DATA SET (EDIT FILE LOCATIONS IF THEY ARE NOT ON THE SAME LEVEL AS THIS PYTHON SCRIPT)
print("Importing data...")
results = pd.read_csv(os.path.join("studentInfo.csv"))
vle = pd.read_csv(os.path.join("studentVle.csv"))
grades = pd.read_csv(os.path.join("studentAssessment.csv"))
assessments = pd.read_csv(os.path.join("assessments.csv"))

#If you wish to supply your own data, import the csv (you will have to move the pandas import line from below, higher) with all the required data columns here to a variable called X_test and set using_own_data = 1
#   If you select this option then no stats will be printed but your grades will be exported to a file called "PREDICTED_GRADES.csv"
#       Note: any number other than 0 or 1 will be treated as a 0.
using_own_data = 0
#EXAMPLE: X_test = pd.read_csv(os.path.join("customdata.csv"))


#DATA PREPARATION SECTION

#calculation of "average_score"
print("Preparing data...")
grades = grades.merge(assessments,on="id_assessment",how="left")
grades["weighted_score"] = grades['score']*grades["weight"]/100.0
av_mark = grades.groupby("id_student")["weighted_score"].sum()/(grades.groupby("id_student")["weight"].sum()/100)
results = results.join(av_mark.rename("average_score"),on=["id_student"],how="left") #merge "average_score" into results
results["average_score"]=results["average_score"].fillna(results["average_score"].mean()) #fill in missing data with the mean

#calculation of "sum_clicks"
studentClicksPerModule = vle.groupby(["code_module","code_presentation","id_student"])["sum_click"].count()
results = results.join(studentClicksPerModule,on=["code_module","code_presentation","id_student"],how='left')

#the removal of our unique identifier ["code_module","code_presentation","id_student"], now that all needed data has been merged to results, and cleaning of the unwanted "region" field
ids = results[["code_module","code_presentation","id_student"]]
X0 = results.drop(["code_module","code_presentation","id_student","region"], axis=1)


#ENCODING THE DATA SECTION#

#integer encoding and filling of missing data with mean
X0["gender"] = X0["gender"].replace(["M","F"],[0,1])
X0["disability"] = X0["disability"].replace(["N","Y"],[1,0])
X0["age_band"] = X0["age_band"].replace(["0-35","35-55","55<="],[1,2,3])
X0["imd_band"] = X0["imd_band"].replace(["Oct-20","0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"],[np.nan,0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])
X0["imd_band"] = X0["imd_band"].fillna(X0["imd_band"].mean())
X0["highest_education"] = X0["highest_education"].replace(["No Formal quals","Lower Than A Level","A Level or Equivalent","HE Qualification","Post Graduate Qualification"],[0,1,2,3,4])
X0["sum_click"]=X0["sum_click"].fillna(X0["sum_click"].mean())

#One-Hot-Encoding section
X1 = X0.drop(["highest_education","imd_band","sum_click","average_score","num_of_prev_attempts", "final_result","studied_credits"], axis=1)
X0 = X0.drop(["age_band","disability","gender"], axis=1)
encoder = OneHotEncoder(handle_unknown='ignore')
X0 = X0.join(pd.DataFrame(encoder.fit_transform(X1).toarray()))

#if statement for activating binary classifier by removing withdrawns and distinctions
if binary == 2:
    X0 = X0.loc[X0["final_result"].isin(["Pass","Fail"])]
#separating label from features
y= X0["final_result"]
X0 = X0.drop("final_result",axis=1)
#if statement for activating binary classifier by counting withdrawns as fails and distinctions as passes
if binary == 1:
    y = y.replace(["Withdrawn","Distinction"],["Fail","Pass"])

#if statement to normalize data if using "svc" model
if model == "svc":
    scaler = StandardScaler().fit(X0)
    X0 = scaler.transform(X0)



#TRAINING AND TEST DATA PREPARATION SECTION
if using_own_data == 0:
    X_train, X_test, y_train, y_test = train_test_split(X0, y, test_size=0.20)
else:
    X_train, y_train, = X0, y


#MODEL CREATION AND TRAINING SECTION
if model == "forest":
    classifier = RandomForestClassifier(bootstrap = True, max_depth = 42, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 85, n_estimators = 410)
elif model == "svc":
    classifier = SVC(kernel='rbf', gamma='auto', C=1.0)
else:
    print("INVALID MODEL TYPE: SHOULD BE 'forest' OR 'svc'")
    sys.exit()
print("Training model...")
classifier.fit(X_train, y_train)


#PREDICTION OF THE TEST DATA AND MEASURING OF PERFORMANCE VIA COMPARING REAL LABELS TO PREDICTED LABELS
if using_own_data == 0:
    print("Training complete! Predicting results and calculating statistics now...\n")
    y_pred = classifier.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    if export_predicted == 1:
        if os.path.exists("PREDICTED_GRADES.csv"):
            os.remove("PREDICTED_GRADES.csv")
        y_pred = pd.DataFrame(y_pred)
        output=y_test.reset_index().merge(y_pred,left_index=True, right_index=True)
        output=ids.merge(output, how="right", left_index=True, right_on="index").drop("index",axis=1).rename(columns={0:"predicted_result"})
        print("\nCompleted! All students' modules' final and predicted results have been saved to 'PREDICTED_GRADES.csv'!\n")
        output.to_csv("PREDICTED_GRADES.csv")
else:
    y_pred = classifier.predict(X_test)
    if os.path.exists("PREDICTED_GRADES.csv"):
            os.remove("PREDICTED_GRADES.csv")
    pd.DataFrame(y_pred).to_csv("PREDICTED_GRADES.csv")
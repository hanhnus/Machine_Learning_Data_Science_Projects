import datetime
import random
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from time                          import time, localtime, strftime
from sklearn.model_selection       import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics               import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, classification_report, precision_recall_curve, roc_curve
from sklearn.metrics               import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing         import StandardScaler
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.svm                   import SVC	
from sklearn.tree                  import DecisionTreeClassifier
from sklearn.ensemble              import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network        import MLPClassifier
from sklearn.linear_model          import LogisticRegression

###########################################################
#               Step 0 - Parameters Setting               #
###########################################################
seed = 309                                                                
random.seed(seed)                      
np.random.seed(seed)   

###########################################################
#                   Step 1 - Load Data                    #
###########################################################
# Function: 1. to load training data & test data from files
#           2. add columns labels 
# Return:   adult_data & adult_test DataFrames
def load_data():                                  
    adult_data = pd.read_csv("../data/adult.data") 
    adult_test = pd.read_csv("../data/adult.test")
    columns    = ["Age", "Workclass", "Final_Weight", "Education", "Highest_Grade", "Marital_Status", "Occupation", 
    			  "Relationship", "Race", "Sex", "Capital_Gain", "Capital_Loss", "Hours_per_Week", "Native_Country", "Income_Class"]
    adult_data.columns = columns
    adult_test.columns = columns
    return adult_data, adult_test  

###########################################################
#             Step 2 - Initial Data Analysis              #
###########################################################
# Function: to get some intuitions about the training data and test data
def initial_data_analysis(df):  
	print("First 5 rows of raw data:")                                
	print(df.head())
	print("Last 5 rows of raw data:")  
	print(df.tail())
	print("Shape of raw data:")
	print(df.shape)                            # (32560, 15)
	print("Any missing data:") 
	print(df.isnull().values.any())            # False
	print("Data Statistical Description:") 
	print(df.describe()) 
	print("Data Types Information:") 
	print(df.info()) 
	print("Check Columns:") 
	print(df.columns)                        # Header added
	for column in df.columns:
		print(column, ':', df[column].unique())


###########################################################
#        Step 3.1 - Data Preprocessing (Cleansing)        #
###########################################################
# Function: to clean the outliers & confusing instances 
# Return:   DataFrame
def data_cleansing(df):  
	df = df[~((df.Relationship == ' Husband') & (df.Sex == ' Female'))]
	df = df[~((df.Relationship == ' Wife')    & (df.Sex == ' Male'))]
	df = df[~(df.Native_Country == ' Holand-Netherlands')]
	return df


###################################################################
# Step 3.2: Data Preprocessing (Feature Selection & Optimization) #
###################################################################
# Function: 1. drop the 'Final_Weight' & 'Education' columns
#           2. combine 'Husband' and 'Wife' to 'Husband-wife' 
#           3. binarize the Income_Class
# Return:   DataFrame
def data_preprocess_feature(df):
	df.drop('Final_Weight', axis = 1, inplace = True)
	df.drop('Education',    axis = 1, inplace = True)

	def combine_features(Relationship):
		if Relationship == ' Husband':
			return 'Husband-wife'
		if Relationship == ' Wife':
			return 'Husband-wife'
		if Relationship == ' Not-in-family':
			return 'Not-in-family'
		if Relationship == ' Other-relative':
			return 'Other-relative'
		if Relationship == ' Own-child':
			return 'Own-child'
		if Relationship == ' Unmarried':
			return 'Unmarried'
	df['Relationship']   = df.apply(lambda row: combine_features(row['Relationship']) , axis = 1)

	def binarize(Income_Class):
		if Income_Class == ' >50K' or Income_Class == ' >50K.':
			return 1
		if Income_Class == ' <=50K' or Income_Class == ' <=50K.':
			return 0
	df['Income_Class']   = df.apply(lambda row: binarize(row['Income_Class']) , axis = 1)

	return df


###########################################################
#       Step 3.3: Data Preprocessing - Dummy Feature      #
###########################################################
# Function: to convert string variables to numerical dummy variables
# Return:   Training set & Test set DataFrames
def data_preprocess_dummy_feature(df): 
	category_list = ["Workclass", "Marital_Status", "Occupation", "Relationship", "Race", "Sex", "Native_Country"]
	# Create dummy variables for categorical data (binary encoding)
	for category in category_list:
		df_dummy = pd.get_dummies(df[category], prefix = category)
		df       = df.join(df_dummy)
	# Determine the categorical columns
	category_dummies_list = df.columns.values.tolist()  # lenth 98
	final_category_list   = [i for i in category_dummies_list if i not in category_list]
	# Remove categorical columns
	df_final         = df[final_category_list]
	# split the labels and data
	df_final_vars    = df_final.columns.values.tolist()
	y_category_list  = ['Income_Class']
	Xs_category_list = [i for i in df_final_vars if i not in y_category_list]
	Xs = df_final[Xs_category_list]          # 90 columns
	y  = df_final['Income_Class']
	return Xs, y


###########################################################
#          Step 3.4: Data Preprocessing - Scaling         #
###########################################################
# Function: to standardize
def data_preprocess_scaling(Xs_train_set, y_train_set, Xs_test_set, y_test_set):
	scaler = StandardScaler()
	scaler.fit(Xs_train_set) 
	Xs_train_set = scaler.transform(Xs_train_set)     
	Xs_test_set  = scaler.transform(Xs_test_set)     
	return Xs_train_set, y_train_set, Xs_test_set, y_test_set


#####################################################################
#    Step 4 - Classification Models Building & Models Assessment    #
#####################################################################
# Function: 1. fit the model
#           2. predict the test set
#           3. calculate accuracy, precision, recall, f1_score, auc
def modelling(model):
	# Construct Model
	model.fit(Xs_train_set, y_train_set)
	# Testing
	y_test_pred = model.predict(Xs_test_set)
	accuracy  = accuracy_score(y_test_set, y_test_pred)  
	precision = precision_score(y_test_set, y_test_pred)
	recall    = recall_score(y_test_set, y_test_pred)
	f1        = f1_score(y_test_set, y_test_pred)
	auc       = roc_auc_score(y_test_set, y_test_pred)
	print('Accuracy:  {var:.4f}'.format(var = accuracy))  
	print('Precision: {var:.4f}'.format(var = precision))
	print('Recall:    {var:.4f}'.format(var = recall))
	print('F1_Score:  {var:.4f}'.format(var = f1))
	print('AUC:       {var:.4f}'.format(var = auc))
	return accuracy, precision, recall, f1, auc


def	save_df_to_csv(df):
	current_time = strftime('%Y-%m-%d_%H-%M-%S', localtime())
	temp_file_name = "part2_temp_@" + str(current_time) + ".csv"
	df.to_csv("part2.csv", encoding="utf_8_sig")
	df.to_csv(temp_file_name, encoding="utf_8_sig")


def	save_tuning_result_to_csv():
	current_time = strftime('%Y-%m-%d_%H-%M-%S', localtime())
	temp_file_name = 'part2_tuning_temp_@' + str(current_time) + '.csv'
	df_tuning.to_csv('part2_tuning.csv', encoding='utf_8_sig')
	df_tuning.to_csv(temp_file_name, encoding='utf_8_sig')


if __name__ == '__main__':

    # Step 1: Load Data
	adult_data, adult_test = load_data()

	# Step 2: Initial Data Analysis
#	initial_data_analysis(adult_data)                     # shape(32560, 15)
#	initial_data_analysis(adult_test)                     # shape(16280, 15)

	# Step 3.1: Data Preprocessing - Cleansing
	adult_data = data_cleansing(adult_data)               # shape(32556, 13)


	# Step 3.2: Data Preprocessing - Feature Selection & Optimization
	adult_data = data_preprocess_feature(adult_data)      # shape(32556, 13)
	adult_test = data_preprocess_feature(adult_test)      # shape(16280, 13)

	# Step 3.3: Data Preprocessing - Dummy Feature  
	Xs_adult_data, y_adult_data = data_preprocess_dummy_feature(adult_data)      # shape(32556, 89) & (32556, 1)   
	Xs_adult_test, y_adult_test = data_preprocess_dummy_feature(adult_test)      # shape(16280, 89) & (16280, 1)

	# Step 3.4: Data Preprocessing - Scaling
	Xs_train_set, y_train_set, Xs_test_set, y_test_set = data_preprocess_scaling(Xs_adult_data, y_adult_data, Xs_adult_test, y_adult_test)


	# Step 4 - Classification Models Building & Models Assessment 
	df_tuning = pd.DataFrame(columns = ['Algorithm', 'para1', 'para2', 'para3', 
		'para4', 'para5', 'para6', 'para7', 'para8', 'para9', 'para10',
		'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC'])
	index = 0
	for model_i in range(7, 8):	
		index += 1	
		if model_i == 1: # p
			print('KNeighborsClassifier')
			for n_neighbors in range(1, 61, 5):
				for weights in ['uniform', 'distance']:
					for leaf_size in range(1, 61, 5):
						for metric in ['manhattan']: # 'chebyshev', 'euclidean', , 'minkowski'
							index += 1	
							model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, leaf_size=leaf_size, metric=metric, n_jobs = -1)
							accuracy, precision, recall, f1, auc = modelling(model)
							df_tuning.loc[index] = ['KNeighborsClassifier', n_neighbors, weights, leaf_size, metric, '', '', '', '', '', '', accuracy, precision, recall, f1, auc]
		if model_i == 2:
			print('GaussianNaiveBayes')
			model = GaussianNB()
			accuracy, precision, recall, f1, auc = modelling(model)
			df_tuning.loc[index] = ['GaussianNaiveBayes', '','', '', '', '', '', '', '', '', '', accuracy, precision, recall, f1, auc]
		if model_i == 3:
			print('SVMClassifier')
			for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
				for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
					for shrinking in [False, True]:
						index += 1
						model = SVC(C = C, kernel = kernel, shrinking = shrinking)
						accuracy, precision, recall, f1, auc = modelling(model)
						df_tuning.loc[index] = ['SVMClassifier', C, kernel, shrinking, '', '', '', '', '', '', '', accuracy, precision, recall, f1, auc]					
		if model_i == 4:
			print('DecisionTreeClassifier')	
			for criterion in ['gini', 'entropy']:
				for splitter in ['best', 'random']:
					index += 1
					model = DecisionTreeClassifier(criterion = criterion, splitter = splitter)
					accuracy, precision, recall, f1, auc = modelling(model)
					df_tuning.loc[index] = ['DecisionTreeClassifier', criterion, splitter, '', '', '', '', '', '', '', '', accuracy, precision, recall, f1, auc]			
		if model_i == 5:
			print('RandomForestClassifier')
			for n_estimators in [10, 50, 100, 500, 1000]:
				for criterion in ['gini', 'entropy']:
					for bootstrap in [False, True]:
						index += 1
						model = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, bootstrap = bootstrap, n_jobs = -1)
						accuracy, precision, recall, f1, auc = modelling(model)
						df_tuning.loc[index] = ['RandomForestClassifier', n_estimators, criterion, bootstrap, '', '', '', '', '', '', '', accuracy, precision, recall, f1, auc]			
		if model_i == 6:
			print('AdaBoostClassifier')
			for n_estimators in [10, 50, 100, 500, 1000]:
				for learning_rate in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
					for algorithm in ['SAMME', 'SAMME.R']:
						index += 1
						model = AdaBoostClassifier(n_estimators = n_estimators, learning_rate = learning_rate, algorithm = algorithm)
						accuracy, precision, recall, f1, auc = modelling(model)
						df_tuning.loc[index] = ['AdaBoostClassifier', n_estimators, learning_rate, algorithm, '', '', '', '', '', '', '', accuracy, precision, recall, f1, auc]			
		if model_i == 7:
			print('GradientBoostingClassifier')
			for loss in ['deviance', 'exponential']: 
				for learning_rate in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0]:
					for n_estimators in [10, 50, 100, 500, 1000]:
						for criterion in ['mse']: #'friedman_mse', 'mse', 'mae'
							index += 1
							print(index)
							model = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, criterion=criterion)
							accuracy, precision, recall, f1, auc = modelling(model)
							df_tuning.loc[index] = ['GradientBoostingClassifier', loss, learning_rate, n_estimators, criterion, '', '', '', '', '', '', accuracy, precision, recall, f1, auc]			
		if model_i == 8:
			print('LinearDiscriminantAnalysis')
			for solver in ['svd', 'lsqr']: #, 'eigen'
#				for shrinkage in [None, 'auto']:
					index += 1
					model = LinearDiscriminantAnalysis(solver = solver)
					accuracy, precision, recall, f1, auc = modelling(model)
					df_tuning.loc[index] = ['LinearDiscriminantAnalysis', solver, '', '', '', '', '', '', '', '', '', accuracy, precision, recall, f1, auc]			
		if model_i == 9:
			print('MultilayerPerceptronClassifier')
			accuracy, precision, recall, f1, auc = modelling(model)
			df_tuning.loc[index] = ['MultilayerPerceptronClassifier', '', '', '', '', '', '', '', '', '', '', accuracy, precision, recall, f1, auc]			
			for activation in ['identity', 'logistic', 'tanh', 'relu']:
				for solver in ['lbfgs', 'sgd', 'adam']:
					for learning_rate in ['constant', 'invscaling', 'adaptive']:
						index += 1
						model = MLPClassifier(activation = activation, solver = solver, learning_rate = learning_rate)
						accuracy, precision, recall, f1, auc = modelling(model)
						df_tuning.loc[index] = ['MultilayerPerceptronClassifier', activation, solver, learning_rate, '', '', '', '', '', '', '', accuracy, precision, recall, f1, auc]			
		if model_i == 10:
			print('LogisticRegression')
			for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
				for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
					for multi_class in ['ovr']:
						index += 1
						model = LogisticRegression(C = C, solver = solver, multi_class = multi_class, n_jobs = -1)
						accuracy, precision, recall, f1, auc = modelling(model)
						df_tuning.loc[index] = ['LogisticRegression', C, solver, multi_class, '', '', '', '', '', '', '', accuracy, precision, recall, f1, auc]
		print("......save_tuning_result_to_csv()......")
		save_tuning_result_to_csv()




































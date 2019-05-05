import pandas as pd
import numpy  as np
import datetime
import random
import matplotlib.pyplot as plt
from time import time, localtime, strftime
from utilities.losses        import compute_loss
from utilities.optimizers    import gradient_descent, pso, mini_batch_gradient_descent
from utilities.visualization import visualize_train, visualize_test
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics         import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model    import LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm             import SVR, LinearSVR
from sklearn.neural_network  import MLPRegressor

###########################################################
#               Step 0 - Parameters Setting               #
###########################################################
seed = 309                                # Freeze the random seed                                 
random.seed(seed)                      
np.random.seed(seed)                     
train_test_split_test_size = 0.3        
FMT = '%Y-%m-%d %H:%M:%S.%f'


###########################################################
#                   Step 1 - Load Data                    #
###########################################################
# Function: to load Data from CSV 
# Return:   DataFrame
def load_data():                                  
    df = pd.read_csv('../data/diamonds.csv')      
    return df                                            


###########################################################
#             Step 2 - Initial Data Analysis              #
###########################################################
# Function: to get some intuitions about the data 
def initial_data_analysis(df):  
	print('First 5 rows of raw data:')                                
	print(df.head())
	print('Last 5 rows of raw data:')  
	print(df.tail())
	print('Shape of raw data:')
	print(df.shape)  
	print('Any missing data:') 
	print(df.isnull().values.any())     
	print('Data Statistical Description:') 
	print(df.describe()) 
	print('Data Types Information:') 
	print(df.info()) 


###########################################################
#        Step 3.1 - Data Preprocessing (Cleansing)        #
###########################################################
# Function: 1. drop the index column
#           2. to clean the outliers 
# Return:   DataFrame
def data_cleansing(df):  
	df.drop(df.columns[0], axis = 1, inplace = True)
	df = df[~(df.y >= 20)]
	df = df[~(df.z >= 20)]
	df = df[~(df.y == 0)]
	df = df[~(df.z == 0)]  
	print('Data Statistical Description(after data cleansing):') 
	print(df.describe()) 
	return df


###########################################################
#          Step 3.2 - Data Preprocessing (Feature)        #
###########################################################
# Function: quantify the classification features 
# Return:   DataFrame
def data_preprocess_feature(df):
    df['cut'].replace('Ideal',     100, inplace = True)
    df['cut'].replace('Premium',    90, inplace = True)
    df['cut'].replace('Very Good',  80, inplace = True)
    df['cut'].replace('Good',       70, inplace = True)        
    df['cut'].replace('Fair',       60, inplace = True)
    df['color'].replace('D',       100, inplace = True)
    df['color'].replace('E',        90, inplace = True)
    df['color'].replace('F',        80, inplace = True)
    df['color'].replace('G',        70, inplace = True)
    df['color'].replace('H',        60, inplace = True)
    df['color'].replace('I',        50, inplace = True)
    df['color'].replace('J',        40, inplace = True)
    df['clarity'].replace('IF',    100, inplace = True)  # Internally Flawless
    df['clarity'].replace('VVS1',   90, inplace = True)  # Very Very Slightly Included - level 1
    df['clarity'].replace('VVS2',   80, inplace = True)  # Very Very Slightly Included - level 2
    df['clarity'].replace('VS1',    70, inplace = True)  # Very Slightly Included - level 1
    df['clarity'].replace('VS2',    60, inplace = True)  # Very Slightly Included - level 2
    df['clarity'].replace('SI1',    50, inplace = True)  # Slightly Included - level 1
    df['clarity'].replace('SI2',    40, inplace = True)  # Slightly Included - level 2
    df['clarity'].replace('I1',     30, inplace = True)  # Included - level 1
    return df


###########################################################
#          Step 3.3 - Data Preprocessing (Split)          #
###########################################################
# Function: 1. split raw data into 70% training set & 30% test set
#           2. split training set & test set into Xs & y, respectively
# Return:   6 DataFrames
def data_preprocess_split(df):
	train_set, test_set = train_test_split(df, test_size = train_test_split_test_size)     
	Xs_train_set = train_set.drop(['price'], axis = 1)      # shape (37758, 9)
	y_train_set  = train_set['price']                       # shape (37758, 1)
	Xs_test_set  = test_set.drop(['price'],  axis = 1)      # shape (16182, 9)
	y_test_set   = test_set['price']                        # shape (16182, 1)
	return train_set, test_set, Xs_train_set, y_train_set, Xs_test_set, y_test_set


###########################################################
#           Step 4 - Exploratory Data Analysis            #
###########################################################
# Function: 1. print the correlation table for training set
#           2. show the correlations by scattering figure
def data_preprocess_EDA():
	print(train_set.corr())

	fig, axs = plt.subplots(3, 3, figsize=(18, 10))
	axs = axs.ravel()
	axs[0].scatter(train_set.carat,   train_set.price, alpha = 0.2, s = 1)
	axs[0].set_xlabel('Carat')
	axs[1].scatter(train_set.cut,     train_set.price, alpha = 0.2, s = 1)
	axs[1].set_xlabel('Cut')
	axs[2].scatter(train_set.color,   train_set.price, alpha = 0.2, s = 1)
	axs[2].set_xlabel('Color')
	axs[3].scatter(train_set.clarity, train_set.price, alpha = 0.2, s = 1)
	axs[3].set_xlabel('Clarity')
	axs[4].scatter(train_set.depth,   train_set.price, alpha = 0.2, s = 1)
	axs[4].set_xlabel('Depth')
	axs[5].scatter(train_set.table,   train_set.price, alpha = 0.2, s = 1)
	axs[5].set_xlabel('Table')
	axs[6].scatter(train_set.x,       train_set.price, alpha = 0.2, s = 1)
	axs[6].set_xlabel('x')
	axs[7].scatter(train_set.y,       train_set.price, alpha = 0.2, s = 1)
	axs[7].set_xlabel('y')
	axs[8].scatter(train_set.z,       train_set.price, alpha = 0.2, s = 1)
	axs[8].set_xlabel('z')
	for i in range(9):
		axs[i].set_ylabel('Price')
		axs[i].set_xlim(auto = True)
		axs[i].set_ylim(auto = True)
	plt.tight_layout(rect=[0, 0, 1, 0.95])
	plt.show()

	
###########################################################
#     Step 3.4 - Data Preprocessing (Standardize Data)    #
###########################################################
# Function: 1. standardize training set input
#           2. standardize test set input
def standardize_data(Xs_train_set, Xs_test_set):
	Xs_train_set_mean = Xs_train_set.mean()                          
	Xs_train_set_std  = Xs_train_set.std()
	Xs_train_set      = (Xs_train_set - Xs_train_set_mean) / Xs_train_set_std
	Xs_test_set       = (Xs_test_set  - Xs_train_set_mean) / Xs_train_set_std

	'''
	fig, axs = plt.subplots(3, 3, figsize=(18, 10))
	axs = axs.ravel()
	axs[0].scatter(Xs_test_set.carat,   y_test_set, alpha = 0.2, s = 1)
	axs[0].set_xlabel('Carat')
	axs[1].scatter(Xs_test_set.cut,     y_test_set, alpha = 0.2, s = 1)
	axs[1].set_xlabel('Cut')
	axs[2].scatter(Xs_test_set.color,   y_test_set, alpha = 0.2, s = 1)
	axs[2].set_xlabel('Color')
	axs[3].scatter(Xs_test_set.clarity, y_test_set, alpha = 0.2, s = 1)
	axs[3].set_xlabel('Clarity')
	axs[4].scatter(Xs_test_set.depth,   y_test_set, alpha = 0.2, s = 1)
	axs[4].set_xlabel('Depth')
	axs[5].scatter(Xs_test_set.table,   y_test_set, alpha = 0.2, s = 1)
	axs[5].set_xlabel('Table')
	axs[6].scatter(Xs_test_set.x,       y_test_set, alpha = 0.2, s = 1)
	axs[6].set_xlabel('x')
	axs[7].scatter(Xs_test_set.y,       y_test_set, alpha = 0.2, s = 1)
	axs[7].set_xlabel('y')
	axs[8].scatter(Xs_test_set.z,       y_test_set, alpha = 0.2, s = 1)
	axs[8].set_xlabel('z')
	for i in range(9):
		axs[i].set_ylabel('Price')
		axs[i].set_xlim(auto = True)
		axs[i].set_ylim(auto = True)
	plt.tight_layout(rect=[0, 0, 1, 0.95])
	plt.show()
	'''
	return Xs_train_set, Xs_test_set


###########################################################
#             Data Preprocessing 5 - Modelling            #
###########################################################
# Function: 1. 
def modelling(model):
	model.fit(Xs_train_set, y_train_set)
	y_pred = model.predict(Xs_test_set)
#	print('Coefficients: ', model.coef_)
#	print('Intersept: ', model.intercept_)
	r2   = r2_score(y_test_set, y_pred)	
#	mse  = mean_squared_error(y_test_set, y_pred)
#	rmse = np.sqrt(mean_squared_error(y_test_set, y_pred))
#	mae  = mean_absolute_error(y_test_set, y_pred)
	print('R2:   {R2:.4f}'.format(R2     = r2))  # R2 should be maximize
#	print(str(model), 'MSE:  {MSE:.2f}'.format(MSE   = mse))
#	print(str(model), 'RMSE: {RMSE:.2f}'.format(RMSE = rmse))
#	print(str(model), 'MAE:  {MAE:.2f}'.format(MAE   = mae))

	'''
	fig, axs = plt.subplots(3, 3, figsize=(18, 10))
	axs = axs.ravel()
	axs[0].scatter(Xs_test_set.carat,   y_test_set, alpha = 0.2, s = 1)
	axs[0].set_xlabel('Carat')
	axs[1].scatter(Xs_test_set.cut,     y_test_set, alpha = 0.2, s = 1)
	axs[1].set_xlabel('Cut')
	axs[2].scatter(Xs_test_set.color,   y_test_set, alpha = 0.2, s = 1)
	axs[2].set_xlabel('Color')
	axs[3].scatter(Xs_test_set.clarity, y_test_set, alpha = 0.2, s = 1)
	axs[3].set_xlabel('Clarity')
	axs[4].scatter(Xs_test_set.depth,   y_test_set, alpha = 0.2, s = 1)
	axs[4].set_xlabel('Depth')
	axs[5].scatter(Xs_test_set.table,   y_test_set, alpha = 0.2, s = 1)
	axs[5].set_xlabel('Table')
	axs[6].scatter(Xs_test_set.x,       y_test_set, alpha = 0.2, s = 1)
	axs[6].set_xlabel('x')
	axs[7].scatter(Xs_test_set.y,       y_test_set, alpha = 0.2, s = 1)
	axs[7].set_xlabel('y')
	axs[8].scatter(Xs_test_set.z,       y_test_set, alpha = 0.2, s = 1)
	axs[8].set_xlabel('z')
	for i in range(9):
		axs[i].set_ylabel('Price')
		axs[i].set_xlim(auto = True)
		axs[i].set_ylim(auto = True)
	plt.tight_layout(rect=[0, 0, 1, 0.95])
	plt.show()
	'''
	return r2

def	save_all_Result_to_csv():
	current_time = strftime('%Y-%m-%d_%H-%M-%S', localtime())
	temp_file_name = 'part1_temp_@' + str(current_time) + '.csv'
	df_r2.to_csv('part1.csv', encoding='utf_8_sig')
	df_r2.to_csv(temp_file_name, encoding='utf_8_sig')

if __name__ == '__main__':

    # Step 1: Load Data
	data = load_data()

	# Step 2: Initial Data Analysis
#	initial_data_analysis(data)

	# Step 3.1: Data Preprocessing - Cleansing
	data = data_cleansing(data)

	# Step 3.2: Data Preprocessing - Feature
	data = data_preprocess_feature(data)

	# Step 3.3: Data Preprocessing - Split
	train_set, test_set, Xs_train_set, y_train_set, Xs_test_set, y_test_set = data_preprocess_split(data)

	# Step 4: Exploratory Data Analysis
#	data_preprocess_EDA()

	# Step 3.4: Data Preprocessing - Standardize Data
	Xs_train_set, Xs_test_set = standardize_data(Xs_train_set, Xs_test_set)

	# Step 5 Regression Models Building & Models Assessment
	df_r2 = pd.DataFrame(columns = ['Algorithm', 'para1', 'para2', 'para3', 
		'para4', 'para5', 'para6', 'para7', 'para8', 'para9', 'para10', 'para11', 
		'para12', 'para13', 'para14', 'para15', 'para16', 'para17', 
		'para18', 'para19', 'para20', 'para21', 'R2'])
	index = 0
	for model_i in range(1, 11):		
		if model_i == 1:
			print('LinearRegression')
			for fit_intercept in [False, True]:
				for normalize in [False, True]:
					model = LinearRegression(fit_intercept = fit_intercept, normalize = normalize, n_jobs = -1)
					r2 = modelling(model)
					df_r2.loc[model] = ['LinearRegression', fit_intercept, normalize, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', r2]
		if model_i == 2:
			print('KNeighborsRegressor')
			for n_neighbors in range(1, 20):
				for weights in ['uniform', 'distance']:
					for leaf_size in range(1, 50):
						for p in [1, 2]:
							for metric in ['minkowski', 'euclidean', 'manhattan', 'chebyshev']:
								model = KNeighborsRegressor(n_neighbors = n_neighbors, weights = weights, algorithm = 'auto', leaf_size = leaf_size, p = p, metric = metric, n_jobs = -1)
								r2 = modelling(model)
								df_r2.loc[model] = ['KNeighborsRegressor', n_neighbors,weights, '', leaf_size, p, metric, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', r2]
		if model_i == 3:
			print('Ridge')
			for alpha in range(1, 200):
				alpha = alpha * 0.01
				for fit_intercept in [False, True]:
					for normalize in [False, True]:
						model = Ridge(alpha = 1.0, fit_intercept = fit_intercept, normalize = normalize, tol = 0.001, solver = 'auto')
						r2 = modelling(model)
						df_r2.loc[model] = ['Ridge', alpha, fit_intercept, normalize, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', r2]
		if model_i == 4:
			print('DecisionTreeRegressor')
			for criterion in ['mse', 'friedman_mse', 'mae']:
				for presort in [False, True]:
					model = DecisionTreeRegressor(criterion = criterion, presort = presort)
					r2 = modelling(model)
					df_r2.loc[model] = ['DecisionTreeRegressor', criterion, presort, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', r2]
		if model_i == 5:
			print('RandomForestRegressor')
			for n_estimators in range(1, 20):
				for criterion in ['mse', 'mae']:
					for oob_score in [False, True]:
						index += 1
						model = RandomForestRegressor(n_estimators = n_estimators, criterion = criterion, oob_score = oob_score, n_jobs = -1)
						r2 = modelling(model)
						df_r2.loc[index] = ['RandomForestRegressor', n_estimators, criterion, '', oob_score, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', r2]
		if model_i == 6:
			print('GradientBoostingRegressor')
			for loss in ['ls']: #, 'ls', 'lad', 'huber', 'quantile']:
				for learning_rate in [0.02]:
					for n_estimators in [1000]:    # the bigger the better
						for max_depth in [5]:      
							for criterion in ['mse']:
								model = GradientBoostingRegressor(loss = loss, learning_rate = learning_rate, n_estimators = n_estimators, criterion = criterion, max_depth = max_depth)
								r2 = modelling(model)
								df_r2.loc[index] = ['GradientBoostingRegressor', loss, learning_rate, n_estimators, criterion, max_depth, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', r2]
		if model_i == 7:
			print('SGDRegressor')
			for loss in ['squared_loss', 'squared_epsilon_insensitive']: # 'huber', 'epsilon_insensitive', 
				for penalty in ['none', 'l2', 'l1', 'elasticnet']:
					for alpha in [0.00001, 0.00005, 0.0001, 0.0005, 0.001]:   
						for l1_ratio in [0.05, 0.1, 0.15, 0.2, 0.25]:      
							for epsilon in [0.01, 0.05, 0.1, 0.5, 1]:
								for learning_rate in ['invscaling', 'constant']: #'optimal'
									index += 1
									model = SGDRegressor(loss = loss, penalty = penalty, alpha = alpha, l1_ratio = l1_ratio, epsilon = epsilon, learning_rate = learning_rate)
									r2 = modelling(model)
									df_r2.loc[index] = ['SGDRegressor', loss, penalty, alpha, l1_ratio, epsilon, learning_rate, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', r2]

		if model_i == 8:
			print('SVR')
			for C in [1.0, 0.1, 0.5, 5.0, 10.0, 50.0, 100.0, 500.0]: #[1.0, 0.1, 0.5, 5.0, 10.0, 50.0, 100.0]:
				for coef0 in [0.0]:
					for kernel in ['linear']: #, 'poly', 'rbf', 'sigmoid'
						index += 1
						model = SVR(kernel = kernel, coef0 = coef0, C = C)
						r2 = modelling(model)
						df_r2.loc[index] = ['SVR', kernel, coef0, C, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', r2]
		
		if model_i == 9:
			print('LinearSVR')
			for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
				for loss in ['squared_epsilon_insensitive']: #'epsilon_insensitive', 
					for dual in [True]:
						index += 1
						model = LinearSVR(C = C, loss = loss, dual = dual)
						r2 = modelling(model)
						df_r2.loc[index] = ['LinearSVR', C, loss, dual, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', r2]
		
		if model_i == 10:
			print('MLPRegressor')
			for activation in ['identity', 'logistic', 'tanh', 'relu']:
				for solver in ['lbfgs', 'adam']: #'sgd'
					for learning_rate in ['constant', 'invscaling', 'adaptive']:
						model = MLPRegressor(hidden_layer_sizes=(100, ), activation = activation, solver = solver,  learning_rate = learning_rate)
						r2 = modelling(model)
						df_r2.loc[index] = ['MLPRegressor', '', activation, solver, learning_rate, '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', r2]
	print(df_r2)	
	save_all_Result_to_csv()



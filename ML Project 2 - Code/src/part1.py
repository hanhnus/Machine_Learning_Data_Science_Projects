import random
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from datetime                import datetime
from time                    import time, localtime, strftime, strptime
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

	return Xs_train_set, Xs_test_set


#####################################################################
#      Step 5 - Regression Models Building & Models Assessment      #
#####################################################################
# Function: 1. fit the model
#           2. predict the test set
#           3. calculate r2, mse, rmse, mae
def modelling(model):
	model.fit(Xs_train_set, y_train_set)
	y_pred = model.predict(Xs_test_set)
	r2     = r2_score(y_test_set, y_pred)	
	mse    = mean_squared_error(y_test_set, y_pred)
	rmse   = np.sqrt(mean_squared_error(y_test_set, y_pred))
	mae    = mean_absolute_error(y_test_set, y_pred)
	print('R2:   {R2:.2f}'.format(R2     = r2))  # R2 should be maximize
	print('MSE:  {MSE:.2f}'.format(MSE   = mse))
	print('RMSE: {RMSE:.2f}'.format(RMSE = rmse))
	print('MAE:  {MAE:.2f}'.format(MAE   = mae))



if __name__ == '__main__':

    # Step 1: Load Data
	data = load_data()

	# Step 2: Initial Data Analysis
	initial_data_analysis(data)

	# Step 3.1: Data Preprocessing - Cleansing
	data = data_cleansing(data)

	# Step 3.2: Data Preprocessing - Feature
	data = data_preprocess_feature(data)

	# Step 3.3: Data Preprocessing - Split
	train_set, test_set, Xs_train_set, y_train_set, Xs_test_set, y_test_set = data_preprocess_split(data)

	# Step 4: Exploratory Data Analysis
	data_preprocess_EDA()

	# Step 3.4: Data Preprocessing - Standardize Data
	Xs_train_set, Xs_test_set = standardize_data(Xs_train_set, Xs_test_set)

	# Step 5 Regression Models Building & Models Assessment
	for model_i in range(1, 21):
		# Record current time
		modelling_start = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
		# To iterate the 10 regression algorithms
		# from  1 to 10: with default parameters setting
		# from 11 to 20: after parameters tuning
		if model_i == 1:
			print('LinearRegression[default]')
			model = LinearRegression()
			modelling(model)
			# Printing Attributes of Linear Regression
			print('Coefficients:   ', model.coef_)
			print('Intercept:      ', model.intercept_)
		if model_i == 11:
			print('LinearRegression')
			model = LinearRegression(fit_intercept = True, n_jobs = -1)
			modelling(model)
			# Printing Attributes of Linear Regression
			print('Coefficients:   ', model.coef_)
			print('Intercept:      ', model.intercept_)
		if model_i == 2:
			print('KNeighborsRegressor[default]')
			model = KNeighborsRegressor()
			modelling(model)	
		if model_i == 12:		
			print('KNeighborsRegressor')
			model = KNeighborsRegressor(n_neighbors = 12, weights = 'distance', algorithm = 'auto', leaf_size = 5, p = 1, metric = 'minkowski', n_jobs = -1)
			modelling(model)
		if model_i == 3:
			print('Ridge[default]')
			model = Ridge()
			modelling(model)
			print('Coefficients:   ', model.coef_)
			print('Intercept:      ', model.intercept_)
		if model_i == 13:	
			print('Ridge')				
			model = Ridge(alpha = 1.0, fit_intercept = True, normalize = False, tol = 0.001, solver = 'auto')
			modelling(model)
			print('Coefficients:   ', model.coef_)
			print('Intercept:      ', model.intercept_)
		if model_i == 4:
			print('DecisionTreeRegressor[default]')
			model = DecisionTreeRegressor()
			modelling(model)
			print('Feature Importances: ', model.feature_importances_)
		if model_i == 14:
			print('DecisionTreeRegressor')
			model = DecisionTreeRegressor(criterion = 'mse', presort = False)
			modelling(model)
			print('Feature Importances: ', model.feature_importances_)
		if model_i == 5:
			print('RandomForestRegressor[default]')
			model = RandomForestRegressor()
			modelling(model)
			print('Feature Importances: ', model.feature_importances_)
		if model_i == 15:
			print('RandomForestRegressor')
			model = RandomForestRegressor(n_estimators = 18, criterion = 'mse', oob_score = False, n_jobs = -1)
			modelling(model)
			print('Feature Importances: ', model.feature_importances_)
		if model_i == 6:
			print('GradientBoostingRegressor[default]')
			model = GradientBoostingRegressor()
			modelling(model)
			print('Feature Importances: ', model.feature_importances_)
			#print('Train Score:         ', model.train_score_)
		if model_i == 16:
			print('GradientBoostingRegressor')
			model = GradientBoostingRegressor(loss = 'ls', learning_rate = 0.02, n_estimators = 1000, criterion = 'mse', max_depth = 5)
			modelling(model)
			print('Feature Importances: ', model.feature_importances_)
			#print('Train Score:         ', model.train_score_)
		if model_i == 7:
			print('SGDRegressor[default]')
			model = SGDRegressor()
			modelling(model)
			print('Coefficients:   ', model.coef_)
			print('Intercept:      ', model.intercept_)
		if model_i == 17:
			print('SGDRegressor')
			model = SGDRegressor(loss = 'squared_loss', penalty = 'l2', alpha = 0.0001, l1_ratio = 0.25, epsilon = 1, learning_rate = 'constant')
			modelling(model)
			print('Coefficients:   ', model.coef_)
			print('Intercept:      ', model.intercept_)
		if model_i == 8:
			print('SVR[default]')
			model = SVR()
			modelling(model)
		if model_i == 18:	
			print('SVR')		
			model = SVR(kernel = 'linear', C = 100.0)
			modelling(model)			
		if model_i == 9:
			print('LinearSVR[default]')
			model = LinearSVR()
			modelling(model)
			print('Coefficients:   ', model.coef_)
			print('Intercept:      ', model.intercept_)			
		if model_i == 19:
			print('LinearSVR')
			model = LinearSVR(C = 5.0, loss = 'squared_epsilon_insensitive', dual = True)
			modelling(model)
			print('Coefficients:   ', model.coef_)
			print('Intercept:      ', model.intercept_)			
		if model_i == 10:
			print('MLPRegressor[default]')
			model = MLPRegressor()
			modelling(model)
			print('Loss:             ', model.loss_ )
			print('Iteration Number: ', model.n_iter_ )
			print('Layer Number:     ', model.n_layers_ )
			print('Output Number:    ', model.n_outputs_ )
		if model_i == 20:
			print('MLPRegressor')
			model = MLPRegressor(hidden_layer_sizes=(100, ), activation = 'relu', solver = 'lbfgs',  learning_rate = 'adaptive')
			modelling(model)
		modelling_end   = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
		modelling_time  = datetime.strptime(modelling_end, FMT) - datetime.strptime(modelling_start, FMT)
		print('Modelling Time: ', str(modelling_time)[-9:], 'seconds')
		print('----------')


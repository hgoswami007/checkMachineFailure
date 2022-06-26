import pandas as pd
import matplotlib.pyplot as plt
import pickle


##Reading the csv file
df = pd.read_csv('ai4i2020.csv')

df.head() # Checking the first five rows of dataset.
df.info() # printing the summary of dataset
df.isnull().sum() #Checking the missing value in dataset

# df = df.fillna(method='mean') # Filling null value with mean value

#Droping the column which is not going to use for feature prediction

df.drop(['Product ID','UDI'],axis=1,inplace=True)

# Convert categorial features into numerical
df = pd.get_dummies(df,columns=['Type'],drop_first=True)


#Assign new index to database last index as prediction result

new_column = ['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]',
              'Tool wear [min]','TWF','HDF','PWF','OSF','RNF','Machine failure']

df = df.reindex(columns=new_column)

# Convert all the data into standscaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]


arr = scaler.fit_transform(x)

# check multi-corelation with each features
from statsmodels.stats.outliers_influence import variance_inflation_factor
via_df =  pd.DataFrame()
via_df['vif'] = [variance_inflation_factor(arr,i) for i in range(arr.shape[1])]
via_df['feature'] = x.columns


# convert dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(arr,y,test_size=0.33,random_state=0)

# Train dataset on linearmodel

from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(x_train,y_train)

# check score before add new data
from sklearn.metrics import r2_score
score = r2_score(linear.predict(x_test),y_test)
print(score)

# Saving the model into the local file system
filename = 'finalized_model_for_car.pickle'
pickle.dump(linear,open(filename,'wb'))

# #Prediction using save model
loaded_model = pickle.load(open(filename,'rb'))





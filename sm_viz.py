# link to the app : https://share.streamlit.io/salma-hamdoune/streamlit_file/code.py

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

header = st.container()
dataset = st.container()
# features = st.container()


modelTesting= st.container()
modelTraining = st.container()
# modelTesting= st.container()

# Code executed once
@st.cache
def get_data(filename):
	housing_data = pd.read_csv(filename)
	return housing_data

# @st.cache
def boxplots(df, x_variable,hue, figsize):
  # Prepare data
  fig, axes = plt.subplots(len(sensors)//2, 2, figsize=figsize)
  for sensor, ax in zip(sensors, axes.ravel()):
    df_ = df[df['Sensor']== sensor]
    if hue == None:
      sns.boxplot(x=x_variable, y='S1T', data=df_, ax=ax)
      # Set Title
      ax.set_title(f'{x_variable}-wise Box Plot\n{sensor}', fontsize=18)
      ax.tick_params(axis='x', rotation=90)
    else:
      sns.boxplot(x=x_variable, y='S1T',hue=hue, data=df_, ax=ax)
      # Set Title
      ax.set_title(f'{x_variable}-wise Box Plot\n{sensor}', fontsize=18)
      ax.tick_params(axis='x', rotation=90)
  plt.show()
# def train_reg(data):
# 	reg = LinearRegression()

# 	df_predictors = data[['NOX' ,'INDUS', 'ZN ']].values
# 	Y = data['MEDV'].values.reshape(-1,1)
# 	reg.fit(df_predictors, Y)

# 	train_pred = reg.predict(df_predictors)
# 	return reg, train_pred
# write in header

with header:
	st.title("Visualisation of Soil Moisture variation over different time quantities")

# with dataset:
# 	st.header('Dataset')
# 	# st.text('This is the dataset')
df = get_data('data/all_parts_S1T.csv')
df_vars = get_data('data/df_vars1.csv')

# 	st.write(df.head())
	# st.subheader("Values distributions of MEDV variable")
	# medv_dist =np.histogram(df['MEDV'], bins=15, range=(0,24))[0]  

	# # # pd.DataFrame(df['MEDV'].value_counts())
	# st.bar_chart(medv_dist)



# with features:
# 	st.header('Features')
sensors = df['Sensor'].unique()

with modelTesting:
	st.header("Variances by sensors")
	st.subheader('For each sensor, the values of soil moisture are aggregated by mean either on days, 12 hours, 8 hours or 6 hours. Then the variance of the resuted values is calculated')
	fig = px.bar(df_vars, x="Sensor", y="value",
             color='Variance', barmode='group',
             height=400)
	st.plotly_chart(fig)

with modelTraining:
	st.header('Boxplots by sensors')
	# sensors = df['Sensor'].unique()
	# for sensor in sensors:
	# 	df_ = df[df['Sensor']== sensor].reset_index(drop=True)
	# 	plot = px.box(df_, x="Date", y="S1T", color = "partofday_12", title=sensor)
	# 	st.plotly_chart(plot)
	
	select_variable = st.selectbox('Aggregation by:', options=["Daily", "12 hours", "8 hours", "6 hours"])

	# twelve_col, eight_col , six_col= st.columns(3)

	# twelve_col.header('partofday_12')
	# with twelve_col:

	if select_variable == "12 hours":
		for sensor in sensors:
			df_ = df[df['Sensor']== sensor].reset_index(drop=True)
			plot = px.box(df_, x="Date", y="S1T", color = "partofday_12", title=sensor)
			st.plotly_chart(plot)

	elif select_variable == "8 hours":
		for sensor in sensors:
			df_ = df[df['Sensor']== sensor].reset_index(drop=True)
			plot = px.box(df_, x="Date", y="S1T", color = "partofday_8", title=sensor)
			st.plotly_chart(plot)

	elif select_variable == "6 hours":
		for sensor in sensors:
			df_ = df[df['Sensor']== sensor].reset_index(drop=True)
			plot = px.box(df_, x="Date", y="S1T", color = "partofday_6", title=sensor)
			st.plotly_chart(plot)

	elif select_variable == "Daily":
		for sensor in sensors:
			df_ = df[df['Sensor']== sensor].reset_index(drop=True)
			plot = px.box(df_, x="Date", y="S1T", title=sensor)
			st.plotly_chart(plot)




	# Y = df['MEDV'].values.reshape(-1,1)
	# predictions = train_reg(df)[1]

	# st.subheader('Mean absolute error of the model is: ')
	# st.write(mean_absolute_error(Y, predictions))

	# st.subheader('Mean squared error of the model is: ')
	# st.write(mean_squared_error(Y, predictions))

	# st.subheader('R squared score of the model is: ')
	# st.write(r2_score(Y, predictions))

# with modelTesting:
# 	st.header('Predict')
# 	# inserting predictors by user
# 	st.subheader('Please fill in values according to each feature name')

# 	st.write('NOX')
# 	val_nox = st.number_input(label="NOX value",step=1.,format="%.2f")

# 	st.write('INDUS')
# 	val_indus = st.number_input(label="INDUS value",step=1.,format="%.2f")

# 	st.write('ZN')
# 	val_zn = st.number_input(label="ZN value",step=1.,format="%.2f")


# 	st.subheader('Model prediction')
# 	st.write(train_reg(df)[0].predict(np.array([[val_nox, val_indus, val_zn]]))[0])


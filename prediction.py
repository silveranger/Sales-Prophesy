import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from pandas.tseries.offsets import DateOffset

from pathlib import Path

# Scikit-Learn models:
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor


class PredictionModel:
  def itemDF(df,option):
      df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
      df['date'] = df['date'].dt.strftime('%Y-%m-01')
      df = df.groupby(['date', 'item'])['sales'].sum().reset_index()
      df = df.pivot_table(index="date", columns="item", values="sales", aggfunc='first', fill_value=0)
      f = pd.DataFrame(index=df.index, columns=['sales', 'date'], data=0)
      f.sales = df[option]
      f.date = pd.to_datetime(df.index, format="%Y-%m-%d")
      f.reset_index(drop=True, inplace=True)
      return f
  def runModel(df,option):
    # df = data.copy()
    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
    df['date']=df['date'].dt.strftime('%Y-%m-01')
    df=df.groupby(['date', 'item'])['sales'].sum().reset_index()
    df=df.pivot_table(index="date", columns="item", values="sales", aggfunc='first', fill_value=0)
    # =====================================
    i=option
    # =====================================

    f=pd.DataFrame(index=df.index,columns=['sales','date'],data=0)
    f.sales=df[i]
    f.date=pd.to_datetime(df.index, format="%Y-%m-%d")
    # f['sales'].plot(kind='line', figsize=(8, 4), title='sales')
    # plt.gca().spines[['top', 'right']].set_visible(False)
    mf=f.copy()
    def get_diff(data):
      """Calculate the difference in sales month over month:"""

      data['sales_diff'] = data.sales.diff()
      data = data.dropna()

      # data.to_csv('./stationary_df.csv')

      return data
    xf=get_diff(f)
    # Let's create a data frame for transformation from time series to supervised:
    def built_supervised(data):
        supervised_df = data.copy()

        # Create column for each lag:
        for i in range(1, 13):
            col_name = 'lag_' + str(i)
            supervised_df[col_name] = supervised_df['sales_diff'].shift(i)

        # Drop null values:
        supervised_df = supervised_df.dropna().reset_index(drop=True)

        return supervised_df

    model_df = built_supervised(f)

    future_dates=[model_df.iloc[-2].date+ DateOffset(months=x)for x in range(0,13)]
    future_dates_df=pd.DataFrame(index=range(0,12),columns=model_df.columns,data=0)
    future_dates_df.date=future_dates[1:]

    def train_test_split(data1,data2):
      data1 = data1.drop(['sales','date'], axis=1)
      data2 = data2.drop(['sales','date'], axis=1)

      train , test = data1.values, data2.values

      return train, test

    train, test = train_test_split(model_df,future_dates_df)
    # print(f"Shape of  Train: {train.shape}\nShape of  Test: {test.shape}")

    def scale_data(train_set,test_set):
      """Scales data using MinMaxScaler and separates data into X_train, y_train,
      X_test, and y_test."""

      # Apply Min Max Scaler:
      scaler = MinMaxScaler(feature_range=(-1, 1))
      scaler = scaler.fit(train_set)

      # Reshape training set:
      train_set = train_set.reshape(train_set.shape[0],
                                    train_set.shape[1])
      train_set_scaled = scaler.transform(train_set)

      # Reshape test set:
      test_set = test_set.reshape(test_set.shape[0],
                                  test_set.shape[1])
      test_set_scaled = scaler.transform(test_set)

      X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel() # returns the array, flattened!
      X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()

      return X_train, y_train, X_test, y_test, scaler


    X_train, y_train, X_test, y_test, scaler_object = scale_data(train, test)
    # print(f"Shape of X Train: {X_train.shape}\nShape of y Train: {y_train.shape}\nShape of X Test: {X_test.shape}\nShape of y Test: {y_test.shape}")

    def re_scaling(y_pred, x_test, scaler_obj, lstm=False):
      """For visualizing and comparing results, undoes the scaling effect on predictions."""
    # y_pred: model predictions
    # x_test: features from the test set used for predictions
    # scaler_obj: the scaler objects used for min-max scaling
    # lstm: indicate if the model run is the lstm. If True, additional transformation occurs

      # Reshape y_pred:
      y_pred = y_pred.reshape(y_pred.shape[0],
                              1,
                              1)

      if not lstm:
          x_test = x_test.reshape(x_test.shape[0],
                                  1,
                                  x_test.shape[1])

      # Rebuild test set for inverse transform:
      pred_test_set = []
      for index in range(0, len(y_pred)):
          pred_test_set.append(np.concatenate([y_pred[index],
                                              x_test[index]],
                                              axis=1) )

      # Reshape pred_test_set:
      pred_test_set = np.array(pred_test_set)
      pred_test_set = pred_test_set.reshape(pred_test_set.shape[0],
                                            pred_test_set.shape[2])

      # Inverse transform:
      pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)

      return pred_test_set_inverted

    def prediction_df(unscale_predictions, origin_df):
      """Generates a dataframe that shows the predicted sales for each month
      for plotting results."""

      # unscale_predictions: the model predictions that do not have min-max or other scaling applied
      # origin_df: the original monthly sales dataframe

      # Create dataframe that shows the predicted sales:
      result_list = []
      sales_dates = future_dates
      act_sales = list(origin_df[-13:].sales)

      for index in range(0, len(unscale_predictions)):
          result_dict = {}
          result_dict['pred_value'] = int(unscale_predictions[index][0] + act_sales[index])
          result_dict['date'] = sales_dates[index + 1]
          result_list.append(result_dict)

      df_result = pd.DataFrame(result_list)

      return df_result
    def plot_results(results, origin_df):
      # results: a dataframe with unscaled predictions
      odf=origin_df.copy()
      fig, ax = plt.subplots(figsize=(15,5))
      sns.lineplot(x=odf.date, y=odf.sales, data=odf, ax=ax,
                  label='Original', color='blue')
      sns.lineplot(x=results.date, y=results.pred_value, data=results, ax=ax,
                  label='Predicted', color='red')


      ax.set(xlabel = "Date",
            ylabel = "Sales",
            title = f"Sales Prophecy Prediction ")

      ax.legend(loc='best')

      filepath = Path('./model_output/{model_name}_forecasting.png')
      filepath.parent.mkdir(parents=True, exist_ok=True)
      plt.savefig(f'./model_output/forecasting.png')
    def regressive_model(train_data, test_data, model):
      """Runs regressive models in SKlearn framework. First calls scale_data
      to split into X and y and scale the data. Then fits and predicts. Finally,
      predictions are unscaled, scores are printed, and results are plotted and
      saved."""

      # Split into X & y and scale data:
      X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data,
                                                                  test_data)

      # Run sklearn models:
      mod = model
      mod.fit(X_train, y_train)
      predictions = mod.predict(X_test) # y_pred=predictions
      # Undo scaling to compare predictions against original data:
      origin_df = f
      unscaled = re_scaling(predictions, X_test, scaler_object) # unscaled_predictions
      unscaled_df = prediction_df(unscaled, origin_df)
      plot_results(unscaled_df, origin_df)
      return pd.DataFrame(unscaled_df)
      # Print scores and plot results:
      # get_scores(unscaled_df, origin_df, model_name)

    result_xg = regressive_model(train, test, XGBRegressor(n_estimators=100, max_depth=1, learning_rate=0.6,
                                                        objective='reg:squarederror'))
    result_lr = regressive_model(train, test, LinearRegression())
    result_rf = regressive_model(train, test, RandomForestRegressor(n_estimators=120, max_depth=20))
    result=pd.DataFrame()
    result['date'] = result_xg['date']
    pred=(result_xg['pred_value']+result_lr['pred_value']+result_rf['pred_value'])/3

    pred=round(pred)
    result['pred_value']=pred
    print(result)

    # if(model=='XGBoost'):
    #     result=regressive_model(train, test, XGBRegressor(n_estimators=100,max_depth=1, learning_rate=0.6,objective='reg:squarederror'), 'XGBoost')
    # elif(model == 'Linear Regression'):
    #     result=regressive_model(train, test, LinearRegression(), 'LinearRegression')
    # elif (model == 'Random Forest'):
    #     result = regressive_model(train, test, RandomForestRegressor(n_estimators=120, max_depth=20), 'RandomForest')

    return result

# data = pd.read_csv('C:/Users/Rushikesh/Downloads/tr.csv')
# obj=XGBoostModel.runModel(data)
# df = data.copy()
# df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
# df['date']=df['date'].dt.strftime('%Y-%m-01')
# print(df)

# df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
# df['date']=df['date'].dt.strftime('%Y-%m-01')
# df=df.groupby(['date', 'item'])['sales'].sum().reset_index()
# df=df.pivot_table(index="date", columns="item", values="sales", aggfunc='first', fill_value=0)
#     # =====================================
# i=1
#     # =====================================
#
# f=pd.DataFrame(index=df.index,columns=['sales','date'],data=0)
# f.sales=df[i]
#
# f.date=pd.to_datetime(df.index, format="%Y-%m-%d")
# print(f)
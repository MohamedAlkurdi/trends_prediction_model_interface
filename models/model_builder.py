import numpy as np
import pandas as pd

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

def build_model(country_trend_data_path, regressors, split_date=pd.to_datetime("2016-02-01")):
    regressors_data = []
    data = None
    error_metrics = {
        "MSE":0,
        "MAE":0,
    }
    
    if regressors:
        for regressor in regressors:
            data = pd.read_csv(regressor["path"], parse_dates=["date"])
            data.set_index("date", inplace=True)
            data.rename(columns={"value": regressor["label"]}, inplace=True)
            regressors_data.append(data)

    country_trend_data = pd.read_csv(country_trend_data_path, parse_dates=["date"])
    country_trend_data.set_index("date", inplace=True)

    data = country_trend_data.join(
        regressors_data,
        how="left",
    )
    
    train_subset = data.loc[data.index < split_date].copy()
    test_subset = data.loc[data.index > split_date].copy()
    
    Pmodel_train_subset = train_subset.reset_index() \
        .rename(columns={
            'date':'ds',
            'score':'y'
        })
        
    print("===== next step is initializing the prophet model amogos")
    model = Prophet()
    
    if regressors:
        for regressor in regressors:
            model.add_regressor(regressor["label"])
    print("====== next step is fitting the model.")
    model.fit(Pmodel_train_subset)

    Pmodel_test_subset = test_subset.reset_index() \
        .rename(columns={
            'date':'ds',
            'score':'y' 
        })
    
    print("====== next step is forecasting the future")
    forecasting_result = model.predict(Pmodel_test_subset)
    
    MSE = np.sqrt(mean_squared_error(y_true=test_subset['score'], y_pred=forecasting_result['yhat']))
    MAE = mean_absolute_error(y_true=test_subset['score'], y_pred=forecasting_result['yhat'])
    error_metrics["MSE"] = MSE
    error_metrics["MAE"] = MAE
    print("*****************")
    print("model:",model)
    print("*****************")
    print("forecasting_result:",forecasting_result)
    print("*****************")
    print("error_metrics:",error_metrics)
    
    return model, forecasting_result, error_metrics, test_subset

def predict_future(model=None, future_preiods=358, data=None, regressors=None):
    if not model:
        raise ValueError("Model is not provided. Please provide a trained model.")
    if data is None or regressors is None:
        raise ValueError("Data and regressors must be provided.")

    future = model.make_future_dataframe(periods=future_preiods, freq='d', include_history=False)

    if regressors:
        for regressor in regressors:
            label = regressor['label']
            future[label] = data[label].mean()
            print(f"Adding regressor: {label}")

    forecast = model.predict(future)
    return forecast

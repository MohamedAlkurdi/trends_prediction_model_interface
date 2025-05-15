import sys
from pathlib import Path
import os

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from .models_configuration_data import models_data_center
from .model_builder import build_model

def find_data_object(country,topic):
    country = country.lower()
    topic = topic.lower()
    
    for model in models_data_center:
        if model["country"] == country and model["topic"] == topic:
            return model

def build_country_topic_model(country,topic):
    print("from build_country_topic_model country:",country)
    print("from build_country_topic_model topic:",topic)
    model_data = find_data_object(country,topic)
    
    # Get the project root directory
    project_root = parent_dir
    
    # Fix path by making it relative to project root
    country_trend_data_path = os.path.join(project_root, model_data['country_trend_data_path'].lstrip('/'))
    
    # Fix regressor paths
    regressors = []
    for regressor in model_data['regressors']:
        regressor_copy = regressor.copy()
        regressor_copy['path'] = os.path.join(project_root, regressor['path'].lstrip('/'))
        regressors.append(regressor_copy)

    print("from build_country_topic_model country_trend_data_path:",country_trend_data_path)
    print("from build_country_topic_model regerssors:",regressors)

    model, forecasting_result, error_metrics, test_subset = build_model(country_trend_data_path, regressors)
    
    test_subset_reset = test_subset.reset_index()

    return {
        "forecasting_result": forecasting_result.to_dict(orient="records"),
        "error_metrics": error_metrics,
        "test_subset": test_subset_reset.to_dict(orient="records"),  
    }
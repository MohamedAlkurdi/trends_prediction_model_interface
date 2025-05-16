import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json

import sys
import os

st.set_page_config(layout="wide")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.similarity_values_calculated_with_cosine import cosine_similarity_values
from frontend.components import *
from frontend.ui_data import *
from frontend.code_snippets import *
from data.similarity_data import similarity_data
from models.build_country_trend_model import build_country_topic_model


API_URL = "http://localhost:5001"
CSV_FILES_DIR = os.path.join(os.path.dirname(__file__), "csv_files")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")

original_dataset = pd.read_csv(os.path.join(CSV_FILES_DIR, "primearchive.blogspot.com_detailled-trends_all-countries.csv"))
original_dataset_split_output_example = pd.read_csv(os.path.join(CSV_FILES_DIR, "cleaned_data_Belgium.csv"))

ner_statisctis1 = pd.read_csv(os.path.join(CSV_FILES_DIR, "entity_counts_every_day.csv"))
ner_statisctis2 = pd.read_csv(os.path.join(CSV_FILES_DIR, "traffic_entities_ratio.csv"))

initial_classification_output_example1 = pd.read_csv(os.path.join(CSV_FILES_DIR, "Australia_classification_output.csv"))
initial_classification_output_example2 = pd.read_csv(os.path.join(CSV_FILES_DIR, "Denmark_classification_output.csv"))
initial_classification_output_example3 = pd.read_csv(os.path.join(CSV_FILES_DIR, "Kenya_classification_output.csv"))

clustered_classification_output_example1 = pd.read_csv(os.path.join(CSV_FILES_DIR, "Canada_clustered_classified.csv"))
clustered_classification_output_example2 = pd.read_csv(os.path.join(CSV_FILES_DIR, "Finland_clustered_classified.csv"))
clustered_classification_output_example3 = pd.read_csv(os.path.join(CSV_FILES_DIR, "Nigeria_clustered_classified.csv"))


clustered_classified_data_with_relative_traffic_rates1 = pd.read_csv(os.path.join(CSV_FILES_DIR, "USA_with_relative_traffic_rates.csv"))
clustered_classified_data_with_relative_traffic_rates2 = pd.read_csv(os.path.join(CSV_FILES_DIR, "UK_with_relative_traffic_rates.csv"))
clustered_classified_data_with_relative_traffic_rates3 = pd.read_csv(os.path.join(CSV_FILES_DIR, "SouthAfrica_with_relative_traffic_rates.csv"))


regressor_example = pd.read_csv(os.path.join(CSV_FILES_DIR, "immigration.csv"))

google_trend_data_sample = pd.read_csv(os.path.join(CSV_FILES_DIR, "exmaple_for_streamlit.csv"))

initial_prophet_model_results = pd.read_csv(os.path.join(CSV_FILES_DIR, "initial_training_results.csv"))


def space(height=1):
    value = ""
    for i in range(height):
        value += "<br/>"
    return st.html(value)


def main():

    st.title("🧮 Audience Interest Trend Forecasting Based on Their Geographical Location")

    space(3)
    
    # Add institution info with logo
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("images/university_logo.jpg")
    with col2:
        st.markdown("""
        **Karadeniz Technical University**  
        Faculty of Technology  
        Department of Software Engineering  
        
        **Graduation Project, 2025**  
        **By**: [Mohammed Alkurdi](https://muhammedalkurdiportfolio.vercel.app/en)  
        **Supervisor**: [DR. MUSTAFA HAKAN BOZKURT](https://avesis.ktu.edu.tr/mhakanbozkurt)
        """)
    space(2)
    st.header("📍 Introduction")
    introduction()

    space(2)

    st.subheader("🎯 Project Objectives")
    objectives()

    space(2)

    st.header("📍 CHAPTER 1: Early Data Preprocessing")

    st.subheader("1.1 - Original dataset")
    st.markdown("The dataset below was our starting point and contains data for news snippets collected from around the world, along with information about news interactions, their date, source, and more.")
    st.markdown("Dataset: [Daily Global Trends 2020 on Kaggle](https://www.kaggle.com/datasets/thedevastator/daily-global-trends-2020-insights-on-popularity)")
    st.write(original_dataset.head())
    
    space() 
    
    st.subheader("✒️ Credit")
    st.markdown("The author of the dataset that I started my project using it is [Jeffrey Mvutu Mabilama.](https://data.world/jfreex)")
    
    space()
    
    st.subheader("1.2 - Origin-based split")
    with st.expander("Code Snippet"):
        Original_Dataset_Splitting_Loop_code_snippet()

    space()

    st.subheader("1.2.1 - Output example")
    st.write(original_dataset_split_output_example.head())

    space()

    st.subheader("1.3 - English only")
    st.markdown(
    """
    After splitting the countries, I decided to target only the English data since the Natural Language Processing tool supports English perfectly. It turned out that there are 12 countries that publish their news in English, and fortunately, each group of three belongs to regions that are either geographically or culturally close.
    """)

    region_countries_dataframe()

    space(2)

    st.header("📍 CHAPTER 2: Data Exploration")

    st.subheader("2.1 - Named Entity Recognition")
    st.markdown("My first experience with data analysis was with the 'Named Entity Recognition' technique, which gave me a morale boost after seeing its results. This provided a good overview, but it became clear that this process was not sufficient.")
    with st.expander("Code Snippet"):
        Named_Entitiy_Recognition_Setup_code_snippet()

    space()

    st.markdown("In this experiment, I studied the relationship between the interaction rate and the 'named entities' on the one hand, and extracted the relationship between time and the reaction rate on the other hand.")

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(ner_statisctis1, height=400)
        st.image(os.path.join(IMAGES_DIR, "visualize_entity_count_everyday.png"))

    with col2:
        st.dataframe(ner_statisctis2, height=400)
        st.image(os.path.join(IMAGES_DIR, "visualize_traffic_entities_ratio.png"))

    space()

    st.subheader("2.2 - Topic Modelling")
    topic_modelling_introduction()

    st.markdown("### 2.2.1 - USING LDA")

    with st.expander("Code Snippet"):
        LDA_Model_Setup_code_snippet()

    col1, col2 = st.columns(2)

    with col1:
        st.image(os.path.join(IMAGES_DIR, "lda_model_vis1.png"))

    with col2:
        st.image(os.path.join(IMAGES_DIR, "lda_model_vis_2.png"))

    space()

    st.markdown("### 2.2.2 - USING BERTopic")

    with st.expander("Code Snippet"):
        BERTopic_Model_Setup_code_snippet()

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            os.path.join(IMAGES_DIR, "newplot.png"),
            use_container_width=True,
        )

    with col2:
        st.image(
            os.path.join(IMAGES_DIR, "newplot2.png"),
            use_container_width=True,
        )

    space(2)

    st.header("📍 CHAPTER 3: Critical Redirection")

    st.subheader("3.1 - Zero-shot Classification")
    space()
    st.subheader("3.1.1 - Initial Classification")
    
    initial_classification_introduction()

    space()

    st.markdown("Using Hugging Face zero-shot models: [Link](https://huggingface.co/tasks/zero-shot-classification)")
    st.markdown("Classification Categories:")
    classification_categories()
    st.markdown("How Classification Model Works:")
    zero_shot_classification_example()
    
    
    with st.expander("Code Snippet"):
        zero_shot_classification_model_workflow_code_snippet()

    st.subheader("3.1.1.1 - Output examples")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(initial_classification_output_example1.head())

    with col2:
        st.write(initial_classification_output_example2.head())

    with col3:
        st.write(initial_classification_output_example3.head())

    space()

    st.subheader("3.1.2 - Category Clustering")
    st.markdown("In the first classification process, the data was classified into 22.")
    st.markdown("However, the number of categories was large for the relatively small amount of data, and therefore the data belonging to each category was too limited to be fully utilized. Therefore, I decided to group the categories into more general categories, thus grouping similar categories and the data belonging to them under one general category that represents all the categories to which they belong.")
    space()
    st.markdown("The category grouping process was carried out as follows:")
    
    categories_clusters_table()
    
    with st.expander("Code Snippet"):
        Clustering_the_detailed_categories_code_snippet()

    st.subheader("3.1.2.1 - Output examples")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(clustered_classification_output_example1.head())

    with col2:
        st.write(clustered_classification_output_example2.head())

    with col3:
        st.write(clustered_classification_output_example3.head())

    st.markdown("Now, that how a topic-country trend might look like:")
    st.image(os.path.join(IMAGES_DIR, "initial_classification_output_visual.png"))

    space()

    st.subheader("3.1.3 - Preprocessing the classified data")
    Preprocessing_the_classified_data_introduction()


    with st.expander("Code Snippet"):
        Normalization_code_snippet()
    
    st.subheader("3.1.3.1 - Output examples")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(clustered_classified_data_with_relative_traffic_rates1.head())

    with col2:
        st.write(clustered_classified_data_with_relative_traffic_rates2.head())

    with col3:
        st.write(clustered_classified_data_with_relative_traffic_rates3.head())

    st.markdown("Here is the visualization of normalized data:")
    st.image(os.path.join(IMAGES_DIR, "general_labeled_data_with_relative_traffice_rate_visual_example.png"))

    space()

    st.subheader("3.2 - Region Grouping")
    region_grouping()
    
    st.subheader("3.2.1 - First similarity calculation method")
    
    with st.expander("Code Snippet"):
        Calculating_similarity_by_direct_comparison_code_snippet()
    
    with st.expander("Intially calculated similiarty values"):
        st.json(similarity_data)
    
    st.subheader("3.2.2 - Updated similarity calculation method")
    
    with st.expander("Code Snippet"):
        cosine_similarity_code_snippet()
    
    regions = list(cosine_similarity_values.keys())
    topics = list(next(iter(cosine_similarity_values.values())).keys())  # get topic list from any region

    selected_region = st.selectbox("🌍 Select a Region", regions)
    selected_topic = st.radio("🧮 Select a Topic", topics, horizontal=True)

    df = cosine_similarity_values[selected_region][selected_topic]
    st.dataframe(df)
    
    space(2)
    
    st.header("📍 CHAPTER 4: Handling the Lack of Data")

    st.subheader("4.1 - Fake Data Generator")
    st.markdown("The initial plan to solve the data shortage problem was to generate data similar to the original data and expand the database accordingly. However, initial experiments showed that this approach suffers from several drawbacks, such as difficulty in controlling the method of generating similar data while maintaining its similarity to reality, or generating data with patterns completely different from the original data.")
    
    with st.expander("Code Snippet"):
        real_data_alike_fake_data_generator_code_snippet()

    st.subheader("4.2 - Google Trends")
    
    google_trends_introduction()

    google_trend_data_sample['date'] = pd.to_datetime(google_trend_data_sample['date'])
    google_trend_data_sample.set_index('date', inplace=True)

    st.markdown("Google trends data sample where it represents the entertainment trend in USA for the last month:")

    st.line_chart(google_trend_data_sample)

    space(2)
    
    st.header("📍 CHAPTER 5: Transformers Time Series Forecasting Models")

    st.subheader("5.1 - First attampt: Keras Model")
    st.markdown("Using the Keras Sequential model with LSTM and Dense layers, I tried to train the Time Forecasting Model using only the original data, but the results were very poor.")
    
    with st.expander("Code Snippet"):
        transformenrs_model_setup_code_snippet()
        
    st.subheader("5.1.1 - Output example")

    st.image(os.path.join(IMAGES_DIR, "transformers_output.png"))
    
    space()
    st.subheader("5.2 - Second attampt: Logic Update")
    st.markdown("I used the extended data later to train the same models. The results got better, but I hypothesized that more context-aware models might yield stronger performance, which led me to explore Prophet later.")
    
    with st.expander("Code Snippet"):
        Second_attampt_code_snippet()

    st.subheader("5.1.2 - Output example")
    st.image(os.path.join(IMAGES_DIR, "transformers_output3.png"))
    
    space(2)
    
    st.header("📍 CHAPTER 6: Facebook Prophet Model")

    st.subheader("6.1 - Model Setup")
    st.markdown("After a quick research, I found out that the Facebook Prophet Time Series Forecasting model might be a good choice. So, I immediately followed a tutorial and implemented a basic logic to test the performance using the extended data.")
    
    with st.expander("Code Snippet"):
        facebook_prophet_model_setup_code_snippet()

    st.subheader("6.1.1 - Output examples")
    example =initial_prophet_model_results.drop("with_regressors_MAE",axis=1)
    st.write(example.head())
    
    space()
    
    st.subheader("6.2 - Regressors")
    regressors_introduction()
    
    regressor_example['date'] = pd.to_datetime(regressor_example['date'])
    regressor_example.set_index('date', inplace=True)
    st.line_chart(regressor_example)
    
    
    st.subheader("6.2.1 - Output examples")
    st.markdown("Optimized performance after adding regressors")
    st.write(initial_prophet_model_results.head())
    space()
    
    st.subheader("6.3 - Country-Topic Trend Prediction Models")
    st.markdown("Here, I manually created 12 to 15 models that predict a topic trend behavior in some country. During this process, I took care of each model and optimized the models that had poor performance, and I succeeded in most cases.")
    
    with st.expander("Model Example"):
        single_prophet_model_example_code_snippet()

    st.subheader("6.3.1 - Output example")
    st.markdown("Below you can see the combination of the original topic-country trend data with the regressors data")
    st.image(os.path.join(IMAGES_DIR, "output.png"))

    st.subheader("6.3.2 - Output example")
    st.markdown("The fllowing visual shows train-test data split information with the predicted values")
    st.image(os.path.join(IMAGES_DIR, "output2.png"))
    st.warning("You can clearly see how distinct the trend data is, yet the prediction results are considered very good because the model has been supported with regressors to enhance the context. This means that the model is not only past-based forecasting model, but rather is an intelligent model.")
    
    st.subheader("6.3.3 - Output example")
    st.markdown("Here is a closer look at a single month trend forecasting.")
    st.image(os.path.join(IMAGES_DIR, "output4.png"))
    
    st.subheader("6.3.4 - Output example")
    st.markdown("Take a look at the components that play a role in producing the final results.")
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(IMAGES_DIR, "output5.1.png"))
        
    with col2:
        st.image(os.path.join(IMAGES_DIR, "output5.2.png"))
    
    st.subheader("6.4 - Country-Topic Model Builder")
    st.markdown("Evantually, I decided to make the model-building process dynamic by writing a script with the required tools for creating a builder depending on the input data. So, if I want to add a new country-topic trend prediction model, all I need to do is add the required data to the data source that the model builder fetches data from.")
    with st.expander("Code Snipprt"):
        dynamic_model_builder_code_snippet()

    space(2)

    st.header("📍 CHAPTER 7: Demo")
    
    space()

    st.subheader("🌏 Select a Country")

    supported_countries = ['usa', 'Canada', 'Australia']
    available_topics = ['entertainment', 'intellectualism', 'nutrition',"politics","economics"]
    
    # Create a dataframe with country names
    world_data = px.data.gapminder().query("year==2007")

    # Add a column to indicate if country is supported
    world_data['is_supported'] = world_data['country'].isin(supported_countries)

    # Create the choropleth map
    fig = px.choropleth(
        world_data,
        locations='iso_alpha',
        color='is_supported',
        hover_name='country',
        color_discrete_map={True: '#00CC96', False: '#A0A0A0'},
        projection='natural earth',
        title='Available Countries',
        labels={'is_supported': 'Supported'}
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_showscale=False,
        geo=dict(
            showframe=True,
            showcoastlines=True,
            projection_type='equirectangular'
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    space()
    selected_country = st.selectbox(
        "Select a country", 
        options=supported_countries,
        index=0
    )
    selected_topic = st.selectbox(
        "Select a topic", 
        options=available_topics,
        index=0
    )

    if selected_country:
        st.success(f"You selected: {selected_country}, {selected_topic}")
    
    
    try:
        result = build_country_topic_model(selected_country, selected_topic)
        forecasting_result = pd.DataFrame(result['forecasting_result'])
        forecasting_result['type'] = 'Forecasted'
        error_metrics = result['error_metrics']
        test_subset = pd.DataFrame(result['test_subset'])
        test_subset['type'] = 'Actual'

        space()
        st.subheader("📊 Error Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Squared Error (MSE)", round(error_metrics['MSE'], 2))
        with col2:
            st.metric("Mean Absolute Error (MAE)", round(error_metrics['MAE'], 2))
            
        mae_value = error_metrics['MAE']
        if mae_value < 8:
            quality = "Excelent"
            quality_color = "green"
        elif mae_value >= 8 and mae_value <= 15:
            quality = "Good"
            quality_color = "orange"
        else:
            quality = "Not Good"
            quality_color = "red"
            
        st.markdown(f"<h3 style='text-align: left; color: {quality_color};'>Forecasting Quality: {quality}</h3>", unsafe_allow_html=True)
        
        space()
        st.subheader("📋 Forecasting Results")
        st.dataframe(forecasting_result)
        space()
        combined_data = pd.concat([
            forecasting_result[['ds', 'yhat', 'type']].rename(columns={'yhat': 'value'}),
            test_subset[['date', 'score', 'type']].rename(columns={'date': 'ds', 'score': 'value'})
        ])
        st.subheader("📈 Forecast Visualization")
        fig = px.line(
            combined_data,
            x='ds',
            y='value',
            color='type',
            title=f"Forecasted Trend for {selected_country.capitalize()} - {selected_topic.capitalize()}",
            labels={'ds': 'Date', 'value': 'Value'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success("Notice that there might be a variation between the actual and forecasted values, but the trend direction is very similar in almost all the cases.")
        space(2)
    except Exception as e:
        st.error(f"Error: {e}")


    st.header("📍 Future Work")
    space(1)
    st.markdown("""
    While this project provides a solid foundation for forecasting audience interest trends, there are several exciting directions for future research and development:

    - **Real-time Forecasting**: Develop an API that provides continuously updated forecasts as new data becomes available
    - **More Topic Categories**: Add specialized categories in emerging areas of interest.
    - **Longer Time Horizons**: Collect more historical data to enable longer-term forecasting (2+ years)
    - **Additional Countries**: Extend the model to include more countries across different continents
    """)

    space(2)

    st.header("📍 Conclusion")
    space(2)
    conclusion()
    space(2)
    st.header(
        """
            💻 [Source Code](https://github.com/MohamedAlkurdi/Trends_prediction_model)
            """
    )

main()

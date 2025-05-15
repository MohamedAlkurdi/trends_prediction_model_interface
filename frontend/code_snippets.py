import streamlit as st


def Original_Dataset_Splitting_Loop_code_snippet():
    return st.markdown(
        """
        **Original Dataset Splitting Loop:**
        ```python
        for country in countries:
            filtered_df = df[df['country'] == country]  # Filter by country
            if not filtered_df.empty:  # Proceed only if there is data for the country
                filtered_df.loc[:, 'name'] = filtered_df['name'].astype(str).apply(clean_text)
                filtered_df.loc[:, 'relatedKeyword'] = filtered_df['relatedKeyword'].astype(str).apply(clean_text)
                filtered_df.loc[:, 'newsTitle'] = filtered_df['newsTitle'].astype(str).apply(clean_text)
                filtered_df.loc[:, 'newsSnippet'] = filtered_df['newsSnippet'].astype(str).apply(clean_text)

                cleaned_df = filtered_df[['index', 'dayId', 'date', 'name', 'traffic', 'newsTitle', 'newsSnippet']]

                output_path = os.path.join(output_dir, f"cleaned_data_{country}.csv")
                cleaned_df.to_csv(output_path, index=False)
        ```
        """
    )


def Named_Entitiy_Recognition_Setup_code_snippet():
    st.markdown(
        """
        **Named Entitiy Recognition Setup:**
        ```python
        nlp = spacy.load("en_core_web_sm")
        data = pd.read_csv("/csv_files/cleaned_data_USA.csv")

        dataFrame = data[['index', 'date','name','traffic', 'newsTitle', 'newsSnippet']] # added name columns

        entities_data = []

        for _, row in dataFrame.iterrows():
            doc = nlp(row['name'])
            
            entities = [ent.label_ for ent in doc.ents]
            
            entities_data.append({
                'id': row['index'],
                'date': row['date'],
                'traffic': row['traffic'],
                'entities': entities 
            })
        ```
        """
    )


def LDA_Model_Setup_code_snippet():
    return st.markdown(
        """
        **LDA Model Setup:**
        ```python
        lda_model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=30,
            random_state=100,
            update_every=1,
            chunksize=100,
            passes=20,
            alpha="auto"
        )
        ```
        """
    )


def BERTopic_Model_Setup_code_snippet():
    return st.markdown(
        """
        **BERTopic Model Setup:**
        ```python
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            top_n_words=10,
            verbose=True
        )

        topics, probs = topic_model.fit_transform(strings_list)
        topic_model.update_topics(strings_list, n_gram_range=(1,3))
        ```
        """
    )


def zero_shot_classification_model_workflow_code_snippet():
    return st.markdown(
        """
        **Model Work Flow:**
        ```python
            classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        labels = ["bussiness","technology","science","sports","media and entertainment","politics","health","crime","accident","environment","art","literature","tragedy","education","fashion","food","travel","military","real estates","history","religion","celebrity"]
        hypothesis_template = "this text is about {}"

        results = []
        batch_size = 16
        for i in range(0, len(strings_list), batch_size):
            batch = strings_list[i:i+batch_size]
            predictions = classifier(batch, labels, hypothesis_template=hypothesis_template, multi_class=True)
            for j, prediction in enumerate(predictions):
                top_label = prediction['labels'][0]
                top_score = prediction['scores'][0]
                row_data = (dataFrame.iloc[i+j]['date'], dataFrame.iloc[i+j]['traffic'], dataFrame.iloc[i+j]['newsSnippet'], top_label, top_score)
                results.append(row_data)
        ```
        """
    )


def Clustering_the_detailed_categories_code_snippet():
    return st.markdown(
        """
        **Clustering the detailed categories:**
        ```python
        data = pd.read_csv('./classified_data/regions/{some_region}/classified_data/{some_country}_classification_output.csv')
        
        label_cluster_mapping = {
            "business": "Economy",
            "real estates": "Economy",
            "technology": "Technology and Science",
            "science": "Technology and Science",
            "media and entertainment": "Entertainment",
            "arts": "Entertainment",
            "sports": "Entertainment",
            "celebrity": "Entertainment",
            "environment": "Lifestyle",
            "health": "Lifestyle",
            "fashion": "Lifestyle",
            "travel": "Lifestyle",
            "food": "Lifestyle",
            "tragedy": "Accident",
            "crime": "Accident",
            "accident": "Accident",
            "politics": "Geopolitical",
            "military": "Geopolitical",
            "education": "Intellectualism",
            "literature": "Intellectualism",
            "history": "Intellectualism",
            "religion": "Intellectualism",
        }
        
        df["general_label"] = df["predicted_label"].map(label_cluster_mapping)
        df.to_csv('./classified_data/regions/{some_region}/clustered_classified_data/{some_country}_clustered_classified.csv', index=False)
        ```
        """
    )


def Normalization_code_snippet():
    return st.markdown(
        """
        **Normalization:**
        ```python
            def calculate_traffic_rate(value, max):
        if max == 0: 
            return 0.0
        rate = float(value / max)
        epsilon = 1e-9  
        if rate <= 0 + epsilon:
            return 0.0
        elif rate < 0.25 - epsilon:
            return 0.1
        elif rate < 0.5 - epsilon:
            return 1/4
        elif rate < 0.75 - epsilon:
            return 1/2
        else:  
            return 1.0
        
        maxTraffic = df['traffic_numeric'].max()
        df['traffic_rate'] = df['traffic_numeric'].apply(lambda x: calculate_traffic_rate(x, maxTraffic))
        
        specific_category = clusters[n]
        specific_category_data = category_time_distribution[
        category_time_distribution['general_label'] == specific_category]
        ```
        """
    )


def Calculating_similarity_by_direct_comparison_code_snippet():
    return st.markdown(
        """
        **Calculating similarity by direct comparison:**
        ```python
        def compare_traffic_rates(file1_path, file2_path):
            try:
                with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
                    reader1 = csv.reader(file1)
                    reader2 = csv.reader(file2)
                    next(reader1)  
                    next(reader2)  
                    similarity_score = 0
                    total_comparisons = 0
                    for row1, row2 in zip(reader1, reader2):
                        if len(row1) < 3 or len(row2) < 3: 
                            print("Error: one of the rows does not have enough columns")
                            return None
                        try:
                            rate1 = float(row1[2])
                            rate2 = float(row2[2])
                        except ValueError:
                            print("Error: Could not convert rate to a number")
                            return None
                        total_comparisons += 1
                        if rate1 == rate2:
                            similarity_score += 1
                    return similarity_score, total_comparisons
            except FileNotFoundError:
                print("Error: One or both files not found.")
                return None
        ```
        """
    )


def cosine_similarity_code_snippet():
    return st.markdown(
        """
        **Calculating similarity by calculating the cosine of the angle between compared vactors:**
        ```python
        def compute_similarity(df1, df2, label):
            df1_filtered = df1[df1['general_label'] == label]
            df2_filtered = df2[df2['general_label'] == label]
            
            common_dates = set(df1_filtered['date']).intersection(df2_filtered['date'])
            df1_filtered = df1_filtered[df1_filtered['date'].isin(common_dates)]
            df2_filtered = df2_filtered[df2_filtered['date'].isin(common_dates)]
            
            df1_filtered = df1_filtered.sort_values('date').reset_index(drop=True)
            df2_filtered = df2_filtered.sort_values('date').reset_index(drop=True)
            
            if len(df1_filtered) < 2 or len(df2_filtered) < 2:
                return None 
            
            traffic_rate_similarity = 1 - cosine(df1_filtered['traffic_rate'], df2_filtered['traffic_rate'])

            return {
                'traffic_rate_similarity': traffic_rate_similarity,
                'common_days': len(common_dates)
            }
        ```
        """
    )


def real_data_alike_fake_data_generator_code_snippet():
    return st.markdown(
        """
        **Attempting to generate orignal-alike fake data using this script:**
        ```python
        def generate_fake_data(real_data: pd.DataFrame, num_days: int) -> pd.DataFrame:

            real_data["date"] = pd.to_datetime(real_data["date"])

            start_date = real_data["date"].max() + timedelta(days=1)
            end_date = start_date + timedelta(days=num_days - 1)
            date_range = pd.date_range(start=start_date, end=end_date)

            synthetic_data = {
                "date": [],
                "general_label": [],
                "traffic_rate": [],
                "total_traffic": [],
            }

            general_label = real_data["general_label"].iloc[0]

            mean_traffic_rate = real_data["traffic_rate"].mean()
            std_traffic_rate = real_data["traffic_rate"].std()
            mean_total_traffic = real_data["total_traffic"].mean()
            std_total_traffic = real_data["total_traffic"].std()

            for date in date_range:
                seasonal_variation = np.sin(2 * np.pi * (date.dayofyear / 365))
                traffic_rate = np.clip(np.random.normal(mean_traffic_rate + seasonal_variation * 0.1, std_traffic_rate * 0.5),0,1,)
                total_traffic = np.clip(np.random.normal(mean_total_traffic * traffic_rate, std_total_traffic * 0.5),0,None,)

                synthetic_data["date"].append(date)
                synthetic_data["general_label"].append(general_label)
                synthetic_data["traffic_rate"].append(traffic_rate)
                synthetic_data["total_traffic"].append(total_traffic)

            synthetic_df = pd.DataFrame(synthetic_data)
            return synthetic_df
        ```
        """
    )


def transformenrs_model_setup_code_snippet():
    return st.markdown(
        """
        **Model Setup:**
        ```python
        def create_model():
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(30, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            return model
        X_train_country1, y_train_country1 = preprocess_data(cointry_data1)
        model = create_model()
        model.fit(X_train_country1, y_train_country1, epochs=50, verbose=1)

        X_train_country2, y_train_country2 = preprocess_data(cointry_data2)
        model.fit(X_train_country2, y_train_country2, epochs=30, verbose=1)
        
        # I continued to tune the model using more tropic-country trned data.
        
        X_train_country12, y_train_country12 = preprocess_data(cointry_data12)
        model.fit(X_train_country12, y_train_country12, epochs=30, verbose=1)
        
        predictions, actual_data = test_on_time_range(
            model, 
            cointry_data4.set_index('date')['traffic_rate'], 
            start_date='2016-11-28', 
            end_date='2017-05-04'
        )
        ```
        """
    )


def Second_attampt_code_snippet():
    return st.markdown(
        """
        **Model Update:**
        ```python
        X_train, y_train, X_test, y_test, scaler = preprocess_data(data, seq_length)

        model = create_model(seq_length)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

        predictions = model.predict(X_test)
        print("predictions:", predictions)

        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test)
        ```
        """
    )


def facebook_prophet_model_setup_code_snippet():
    return st.markdown(
        """
        **Model Setup:**
        ```python
        train_subset = data.loc[data.index <= split_date].copy()
        test_subset = data.loc[data.index > split_date].copy()
        
        model = Prophet()
        model.fit(Pmodel_train_subset)
        ```
        """
    )


def single_prophet_model_example_code_snippet():
    return st.markdown(
        """
            ```python
        regressors = [
            {
                "label": "diplomacy",
                "path": "/Geopolitical/extended_data/usa_trend_data/regressors/diplomacy.csv",
            },
            {
                "label": "economy",
                "path": "/Geopolitical/extended_data/usa_trend_data/regressors/economy.csv",
            },
            {
                "label": "elections",
                "path": "/Geopolitical/extended_data/usa_trend_data/regressors/elections.csv",
            },
            {
                "label": "war",
                "path": "/Geopolitical/extended_data/usa_trend_data/regressors/war.csv",
            },
            {
                "label": "immigration",
                "path": "/Geopolitical/extended_data/usa_trend_data/regressors/immigration.csv",
            },
        ]
        regressors_data = []

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

        model = Prophet()
        for regressor in regressors:
            model.add_regressor(regressor["label"])
        model.fit(Pmodel_train_subset)
        ```
        """
    )


def dynamic_model_builder_code_snippet():
    return st.markdown(
        """
        ```python
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
        
        # rename the date and target columns to make them suitable for the prophet model
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
        ```
    """
    )

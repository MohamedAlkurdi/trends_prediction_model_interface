import streamlit as st

def introduction():
    return st.markdown(
        """
    The overall goal of this project is attempting to predict the future performance of communities based on geographic location and cultural interests, in order to support better decision-making processes targeting society as a whole.  

    Rather than predicting precise numerical performance, this project focuses on forecasting the fluctuations in public interest toward specific topics within given geographic regions.

    It is important to mention that I was not exactly trying to solve a specific real-world problem, so my work is an extended first step in the field of decision-making support processes using real-world data examples.

    Below, I will take you on a quick journey through the steps of implementing this project. During this journey, you will notice developments in the project's working method, which I hope reflect my ability to deal with unexpected changes.
    """
    )
    
def objectives():
    return st.markdown(
        """
    1. To solve a problem that can be applied in the real world while ensuring its scalability.
    2. To test the feasibility of predicting community behavior and exploiting this knowledge to tailor public-facing actions.
    3. To attempt to predict trends in interest related to a specific topic in a specific geographic area.
    """
    )

def topic_modelling_introduction():
    return st.markdown(
        """
            After noticing the limited results of the 'Named Entity Recognition' process, I decided to try another method that focused on analyzing natural language and extracting the most frequent topics, a process called 'topic modeling'.
            
            In this process, I tried several techniques, modifying the inputs and configuration variables many times in an attempt to achieve the best results.
            
            I first used the LDA algorithm and later used the BERTopic model for the same purpose.
            """
        ""
    )
    
def initial_classification_introduction():
    return st.markdown(
        """
        So far, I was using methods that belong to the "Unsupervised Machine Learning" style, which depends on recognizing patterns and trying to extract an underlying meaning from data. Apparently, this is not the best option.
        So, the next step was to consider taking the "Supervised Learning" approach, which means I would label the data and later train the model on these labels so it could classify inputs after learning.
        To do the "labeling" task, there were a few options:
        1. Labeling the data manually, which can take a massive amount of time.
        2. Using a pretrained classification model that can classify your data, and indeed that is what happened.
        So, I started looking for a pretrained classification model and found one soon, allowing me to start the most important phase in my project.
        The technique used is called "zero-shot classification," and it is defined as "a task in natural language processing where a model is trained on a set of labeled examples but is then able to classify new examples from previously unseen classes".
        """
    )
    
def Preprocessing_the_classified_data_introduction():
    return st.markdown(
        """
            To move to the next step, I had to reshape the data and make it simpler yet more understandable to use it efficiently.
        
            Here is how I reshaped the data:

            1. Normalized the values of topic-interaction rates.
            2. Deleted the unnecessary columns.
            3. The dates were set, and the data was arranged in chronological order daily.
            4. Sub-datasets were distributed based on their subject and country of origin.
                """
    )
    
def region_grouping():
    return st.markdown(
        """
        In the early stages of the project, I anticipated a lack of training data for the prediction model I wanted to train in the later stages of the project. Therefore, I considered the possibility of grouping several countries into a single region and using all the data from each region to predict the future performance of the entire region. I also implemented various procedures to collect as much data as possible under a specific category or country.
        At this stage, I thought the time was right to group the countries into regions. To group the countries, I needed to find a specific criterion to base my work on. So, I decided to calculate the similarity score between the countries' behaviors, compare them, and group the most similar countries together into a single region.
        I conducted this process in two stages. In the first stage, I tried to calculate the similarity scores by directly comparing interaction values, but I got poor results because the method of calculating similarity did not fully align with my idea of calculating the similarity score in overall interaction performance trends.
        """
    )    
    
def google_trends_introduction():
    return st.markdown("""
        After abandoning the previous idea, I searched for new potential solutions and discovered the "Google Trends", which emerged as a game-changing data source, offering highly relevant and structured information that closely mirrored my original dataset, thus enabling a seamless transition to more scalable and reliable modeling. meaning I didn't have to put in extra effort to process the imported data. In addition, the data scope is wide and diverse. However, I didn't start collecting data haphazardly. Instead, I first verified that Google's data was similar to the data I was working with in terms of performance and characteristics. I compared samples of the data I was working with to Google's data and concluded that Google's data was similar to a large portion of my data. Based on this, I made some adjustments, which I can summarize as follows:
        - I abandoned the division of countries into regions since there is ample data for each country and for any topic I wanted.
        - I modified the categories I had defined at the beginning of the project to make them consistent with the data categories on the Google Trends platform.
        - I used a vector similarity calculation method between the original data and the Google data.
        - Collecting data over a period of approximately five years, starting from 2012 and ending with the time period of the original data, in order to use the period of the original data in the process of testing the performance of the forecasting model that will be developed later.
        """) 
    
def regressors_introduction():
    return st.markdown(
        """
                To enhance the performance of the model, I started to import data to enforce the context of the training dataset. This would make the model smarter by noticing unusual events instead of being only a past-data-based prediction model.
                
                This process took me some time because it is not very straightforward. It requires general knowledge about the countries' interests and periodic events and requires me to make some assumptions, test them, and adjust them. However, it eventually paid off.
                
                The chart below visualizes the **immigration** regressor, where I used it to check whether considering this data may enhance the **Geopolitical** trend interactivity forecasting in **USA** or not.
                """
    )

def conclusion():
    return st.markdown(
        """
            The project work journey was enjoyable and challenging. During the process, I learned about my capabilities and was able to build a solid foundation of skills and knowledge that will undoubtedly help me in my future projects. If I want to summarize the benefits of the journey in a few points, I could say:
            1. Plan well before you begin the implementation.
            2. Challenges and problems are not the end of the road; rather, they are an opportunity for learning, growth, and self-discovery.
            3. Try to solve the problem on your own before exploring the optimal solution so you can better understand both the problem and the solution.
                """
    )
    
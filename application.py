import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

# Load the clustering model and scaler
with open('Task_1/kmeans_model.pkl', 'rb') as f:
    kmeans = pkl.load(f)

with open('Task_1/scaler.pkl', 'rb') as f:
    scaler = pkl.load(f)

# Load the classification model
with open('Task_1/rvc_model.pkl', 'rb') as f:
    classifier = pkl.load(f)

with open('Task_1/scaler_clf.pkl', 'rb') as f:
    scaler_clf = pkl.load(f)

with open('Task_1/encoder.pkl', 'rb') as f:
    encoder = pkl.load(f)

# Function to predict the cluster of new data
def prediction(new_data, kmeans_model, scaler_model):
    new_data = scaler_model.transform(new_data)
    prediction = kmeans_model.predict(new_data)
    explanations = []
    for idx, data_point in enumerate(new_data):
        cluster_idx = prediction[idx]
        centroid = kmeans_model.cluster_centers_[cluster_idx]
        distance = np.linalg.norm(data_point - centroid)
        explanation = f"The data point belongs to cluster {cluster_idx} because it is closest to the centroid with a distance of {distance:.2f}."
        explanations.append((prediction[idx], explanation))
    
    return prediction, explanations

# Task 2: Function to train and predict using the classifier
def predict(X):
    X.dropna(axis=0, inplace=True)
    X_scaled = scaler_clf.transform(X)
    
    # Predict on test set
    y_pred_encoded = classifier.predict(X_scaled)
    y_pred = encoder.inverse_transform(y_pred_encoded)
    
    return y_pred

# Task 3: Function to calculate durations and activity counts
def calculate_durations_and_counts(df):
    df.dropna(axis=0, inplace=True)
    df = df.sort_values(by=['date', 'time'])
    df['position'] = df['position'].str.lower()
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    df['duration'] = df.groupby(['date', 'position'])['datetime'].diff().dt.total_seconds().div(60)
    
    inside_data = df[df['position'] == 'inside']
    outside_data = df[df['position'] == 'outside']
    
    inside_duration = inside_data.groupby('date')['duration'].sum().reset_index()
    inside_duration.columns = ['date', 'total_inside_duration']
    
    outside_duration = outside_data.groupby('date')['duration'].sum().reset_index()
    outside_duration.columns = ['date', 'total_outside_duration']
    
    total_duration = pd.merge(inside_duration, outside_duration, on='date', how='outer')
    
    picking_count = df[df['activity'] == 'picked'].groupby('date').size().reset_index(name='picking_count')
    placing_count = df[df['activity'] == 'placed'].groupby('date').size().reset_index(name='placing_count')
    
    activity_counts = pd.merge(picking_count, placing_count, on='date', how='outer')
    
    final_output = pd.merge(total_duration, activity_counts, on='date', how='outer')
    final_output['date'] = final_output['date'].astype(str)
    
    return final_output

# Streamlit app
st.title("Data Processing and Analysis")

task = st.selectbox("Select the task you want to perform", ("Task 1: Clustering", "Task 2: Classification", "Task 3: Duration and Activity Counts"))

if task == "Task 1: Clustering":
    st.header("Task 1: Clustering")
    input_type = st.radio("Choose input method", ("Upload file", "Manually enter data"))

    if input_type == "Upload file":
        uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "csv"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("### Raw Data")
            st.write(df.head())
            
            if st.button('Run Clustering'):
                try:
                    prediction, explanations = prediction(df, kmeans, scaler)
                    df['Cluster'] = prediction
                    st.write("Cluster Predictions:")
                    st.write(prediction)
                    st.write("Explanations:")
                    for exp in explanations:
                        st.write(exp[1])
                except Exception as e:
                    st.error(f"Error in clustering: {e}")

    else:
        data_point = st.text_input('Enter data point for clustering (comma separated values):')
        if st.button('Run Clustering'):
            try:
                data_point = np.array(data_point.split(',')).astype(float).reshape(1, -1)
                prediction, explanations = prediction(data_point, kmeans, scaler)
                st.write("Explanations:")
                for exp in explanations:
                        st.write(exp[1])
            except ValueError:
                st.error("Please ensure that the data point is a comma-separated list of numbers.")
            except Exception as e:
                st.error(f"Error in clustering: {e}")

elif task == "Task 2: Classification":
    st.header("Task 2: Classification")
    input_type = st.radio("Choose input method", ("Upload file", "Manually enter data"))

    if input_type == "Upload file":
        uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "csv"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("### Raw Data")
            st.write(df.head())

            if st.button("Run Classification"):
                try:
                    y_pred = predict(df)
                    df['Prediction'] = y_pred
                    st.write(y_pred)
                    st.write(df)
                except Exception as e:
                    st.error(f"Error in classification: {e}")

    else:
        data_point = st.text_input('Enter data point for classification (comma separated values):')
        if st.button("Run Classification"):
            try:
                data_point = np.array(data_point.split(',')).astype(float).reshape(1, -1)
                data_point_scaled = scaler_clf.transform(data_point)
                prediction = classifier.predict(data_point_scaled)[0]
                st.write(f"Prediction: {encoder.inverse_transform([prediction])[0]}")
            except ValueError:
                st.error("Please ensure that the data point is a comma-separated list of numbers.")
            except Exception as e:
                st.error(f"Error in classification: {e}")

else:
    st.header("Task 3: Duration and Activity Counts")
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "csv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### Raw Data")
        st.write(df.head())

        if st.button("Run Task 3"):
            try:
                result = calculate_durations_and_counts(df)
                st.write(result)
                st.write("Download Results as CSV")
                result_csv = result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=result_csv,
                    file_name='final_output.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Error in calculating durations and activity counts: {e}")

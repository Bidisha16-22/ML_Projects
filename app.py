import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# --- Function to generate download link ---
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href

# --- Main Title ---
st.title("Heart Disease Predictor")
st.write("Enter your details below to know the prediction")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

# --- Prediction Tab ---
with tab1:
    Age = st.number_input("Age (years)", min_value=0, max_value=150)
    Sex = st.selectbox("Sex", ["Male", "Female"])
    ChestPainType = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    RestingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    Cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    FastingBS = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    RestingECG = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
    MaxHR = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    ExerciseAngina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    Oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    ST_Slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    if st.button("Submit"):
        # --- Convert categorical inputs ---
        Sex_num = 0 if Sex == "Male" else 1
        ChestPainType_num = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(ChestPainType)
        FastingBS_num = 1 if FastingBS == "> 120 mg/dl" else 0
        RestingECG_num = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(RestingECG)
        ExerciseAngina_num = 1 if ExerciseAngina == "Yes" else 0
        ST_Slope_num = ["Upsloping", "Flat", "Downsloping"].index(ST_Slope)

        # --- Input DataFrame ---
        input_data = pd.DataFrame({
            'Age': [Age],
            'Sex': [Sex_num],
            'ChestPainType': [ChestPainType_num],
            'RestingBP': [RestingBP],
            'Cholesterol': [Cholesterol],
            'FastingBS': [FastingBS_num],
            'RestingECG': [RestingECG_num],
            'MaxHR': [MaxHR],
            'ExerciseAngina': [ExerciseAngina_num],
            'Oldpeak': [Oldpeak],
            'ST_Slope': [ST_Slope_num]
        })

        algonames = ['Decision Tree','Logistic Regression','Random Forest','Support Vector Machine']
        modelnames = ['tree.pkl', 'LogisticR.pkl', 'RandomF.pkl', 'SVM.pkl']

        st.subheader("Results:")
        st.markdown('-----------------------')

        for i, modelname in enumerate(modelnames):
            with open(modelname, 'rb') as f:
                model = pickle.load(f)
                prediction = model.predict(input_data)[0]
                st.subheader(algonames[i])
                if prediction == 0:
                    st.write("No heart disease detected.")
                else:
                    st.write("Heart disease detected.")
                st.markdown('-------------------------')

# --- Bulk Prediction Tab ---
with tab2:
    st.header("Upload CSV File")
    st.subheader("Instructions:")
    st.info('''
        1. No missing (NaN) values.\n
        2. Must have all 11 features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, 
        RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope.\n
        3. Values must be properly encoded:\n
            - Sex: 0=Male, 1=Female\n
            - ChestPainType: 0=Atypical Angina, 1=Non-Anginal Pain, 2=Asymptomatic, 3=Typical Angina\n
            - FastingBS: 0 or 1\n
            - RestingECG: 0=Normal, 1=ST-T wave Abnormality, 2=Left Ventricular Hypertrophy\n
            - ExerciseAngina: 0=No, 1=Yes\n
            - ST_Slope: 0=Upsloping, 1=Flat, 2=Downsloping
    ''')

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df_bulk = pd.read_csv(uploaded_file)
        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        if set(expected_columns).issubset(df_bulk.columns):
            model = pickle.load(open('LogisticR.pkl', 'rb'))
            predictions = model.predict(df_bulk[expected_columns])
            df_bulk['prediction_LR'] = predictions

            st.subheader("Predictions:")
            st.write(df_bulk)

            csv = df_bulk.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Predictions as CSV",
                               data=csv,
                               file_name='heart_predictions.csv',
                               mime='text/csv')
        else:
            st.warning("Uploaded CSV must contain all required columns.")
    else:
        st.info("Upload a CSV file to get predictions.")

with tab3:
    import plotly.express as px
    data = {'Decision Tree ': 80.97, 'Logistic Regression': 85.86, 'Random Forest': 84.23, 'Support Vector Machine': 84.22}
    Models = list(data.keys())
    Accuracies = list(data.values())
   # df= pd.DataFrame(list(zip(Models, Accuracies )), columns = ["Model", "Accuracy"])
    #fig = px.bar(df,Y='Accuracies', X='Models')
    #st.plotly_chart(fig)
    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=["Model", "Accuracy"])

    fig = px.bar(df, y='Accuracy', x='Model', title="Model Accuracy Comparison", text='Accuracy')
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(yaxis_range=[0, 100])

    st.subheader("Model Performance")
    st.plotly_chart(fig)
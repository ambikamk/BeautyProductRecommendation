import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
model_file_path = 'best_model.pkl'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

# Load the preprocessed data for feature encoding reference
preprocessed_file_path = 'preprocessed_data.csv'
df = pd.read_csv(preprocessed_file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Feature engineering based on preprocessed data
features = ['Skin_Tone', 'Skin_Type', 'Rating', 'Brand', 'Price', 'Ingredients', 'Category']
df_encoded = pd.get_dummies(df[features])
X_train_columns = df_encoded.columns

# Function to recommend product
def recommend_product(input_features):
    input_df = pd.DataFrame([input_features])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X_train_columns, fill_value=0)
    prediction = model.predict(input_encoded)
    return prediction[0]

# Streamlit UI
st.set_page_config(page_title="Beauty Product Recommendation System", layout="wide", initial_sidebar_state="expanded")

st.title("Beauty Product Recommendation System")
st.write("### Find the best beauty products tailored to your needs!")

with st.sidebar:
    st.header("Enter Your Details")
    skin_tone = st.selectbox("Skin Tone", df['Skin_Tone'].unique())
    skin_type = st.selectbox("Skin Type", df['Skin_Type'].unique())
    rating = st.slider("Rating", 0.0, 5.0, step=0.1)
    brand = st.selectbox("Brand", df['Brand'].unique())
    price = st.selectbox("Price", df['Price'].unique())
    ingredients = st.selectbox("Ingredients", df['Ingredients'].unique())
    category = st.selectbox("Category", df['Category'].unique())
    
    input_features = {
        'Skin_Tone': skin_tone,
        'Skin_Type': skin_type,
        'Rating': rating,
        'Brand': brand,
        'Price': price,
        'Ingredients': ingredients,
        'Category': category
    }

    if st.button("Recommend", key="recommend_button"):
        recommended_product = recommend_product(input_features)
        st.session_state['recommended_product'] = recommended_product

# Main section for displaying results
st.write("## Entered Details:")
st.json(input_features)

if 'recommended_product' in st.session_state:
    st.write("## Recommendation:")
    st.write(f"Recommended Product: **{st.session_state['recommended_product']}**")

# Style the app
st.markdown("""
    <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: white;
            color: black;
            border: 2px solid #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

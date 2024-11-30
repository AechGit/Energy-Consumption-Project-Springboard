import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import joblib

class EnergyConsumptionApp:
    def __init__(self):
        st.set_page_config(
            page_title="Energy Insights Pro",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.setup_custom_css()
        self.setup_page()
        self.load_resources()
 
    def setup_custom_css(self):
        st.markdown("""
        <style>
        /* Custom color palette */
        :root {
            --primary-color: #95a5a6;      /* Gray color replacing blue */
            --secondary-color: #3498db;    /* Blue color replacing red */
            --background-color: #f4f6f7;   /* Light background color */
            --text-color: #2c3e50;         /* Dark text color */
            --off-white: #bdc3c7;           /* Gray color replacing off-white */
            --green-color: #2ecc71;        /* Green color for regression texts */
            --blue-color: #3498db;         /* Blue color for input labels */
            --white-color: #ffffff;        /* White color for methodology and disclaimer */
        }

        /* Ensure sidebar is on the side */
        .css-1aumxhk {
            position: fixed;
            top: 0;
            right: 0;
            bottom: 0;
            width: 300px;
            padding: 20px;
            background-color: white;
            box-shadow: -4px 0 6px rgba(0, 0, 0, 0.1);
            overflow-y: auto;
            z-index: 1000;
        }

        /* Main content adjustment */
        .css-1aehpvj {
            margin-right: 300px;
        }

        /* Slider container styling */
        .stSlider {
            background-color: var(--off-white);
            border: 2px solid var(--primary-color);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }

        /* Slider track styling */
        .stSlider > div > div {
            background-color: var(--primary-color);
            color: white;
        }

        /* Global styling */
        body {
            color: var(--text-color);
            background-color: var(--background-color);
            font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
        }

        /* Label styling for blue inputs */
        .stSlider label,
        .stNumberInput label,
        .stSelectbox label,
        .stTextInput label,
        .stRadio label {
            color: var(--blue-color);
        }

        /* Prediction card styling */
        .prediction-card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
        }

        /* Button styling */
        .stButton > button {
            background-color: var(--primary-color);
            color: white;
            border-radius: 5px;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background-color: var(--secondary-color);
            transform: scale(1.05);
        }

        /* Linear and Ridge Regression text styling */
        .linear-regression-text, .ridge-regression-text {
            color: var(--green-color);
        }

        /* Methodology & Disclaimer Text */
        .stExpander .stMarkdown {
            color: var(--white-color) !important;
        }
        .stExpander .stMarkdown h3,
        .stExpander .stMarkdown h4 {
            color: var(--white-color) !important;
        }
        </style>
        """, unsafe_allow_html=True)
 
    def setup_page(self):
        st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #3498db;">⚡ Energy Insights Pro</h2>
        </div>
        <h3>Customize Prediction Parameters</h3>
        """, unsafe_allow_html=True)
 
    def load_resources(self):
        try:
            self.linear_model = joblib.load("linear_model.pkl")
            self.ridge_model = joblib.load("ridge_model.pkl")
            self.feature_names = joblib.load("feature_names.pkl")
            
            st.toast("Models loaded successfully! 🚀", icon="✅")
        except FileNotFoundError as e:
            st.error(f"File not found: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error loading resources: {e}")
            st.stop()
 
    def run(self):
        # Main content area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            # 🔌 Energy Consumption Forecasting
            Predict your energy consumption with advanced machine learning models.
            """)

        # Feature inputs in sidebar
        voltage = st.sidebar.slider("Voltage (V)", 220.0, 255.0, 240.0, help="Adjust the voltage level")
        global_intensity = st.sidebar.slider("Global Intensity (A)", 0.0, 20.0, 4.63, help="Set the electrical current intensity")
        sub_metering_1 = st.sidebar.slider("Sub Metering 1 (Wh)", 0.0, 50.0, 1.12, help="Energy consumption for sub-meter 1")
        sub_metering_2 = st.sidebar.slider("Sub Metering 2 (Wh)", 0.0, 50.0, 1.30, help="Energy consumption for sub-meter 2")
        sub_metering_3 = st.sidebar.slider("Sub Metering 3 (Wh)", 0.0, 50.0, 6.46, help="Energy consumption for sub-meter 3")
 
        # DateTime inputs
        date = st.sidebar.date_input("Select Date", value=pd.Timestamp("2024-11-28"))
        time = st.sidebar.time_input("Select Time", value=pd.Timestamp("2024-11-28 12:00:00").time())
 
        # Derived features
        date_time = pd.Timestamp.combine(date, time)
        year, month, day, hour, minute = (
            date_time.year,
            date_time.month,
            date_time.day,
            date_time.hour,
            date_time.minute,
        )
        is_holiday, light, weekday = 0, 1, date_time.weekday()
 
        # Prepare input data with features
        input_data = pd.DataFrame({
            "Global_reactive_power": [0.0],
            "Voltage": [voltage],
            "Global_intensity": [global_intensity],
            "Sub_metering_1": [sub_metering_1],
            "Sub_metering_2": [sub_metering_2],
            "Sub_metering_3": [sub_metering_3],
            "Year": [year],
            "Month": [month],
            "Day": [day],
            "Hour": [hour],
            "Minute": [minute],
            "Is_holiday": [is_holiday],
            "Light": [light],
            "Weekday": [weekday]
        })
 
        # Match input features to model's expected features
        try:
            input_data = input_data[self.feature_names]
        except KeyError as e:
            st.error(f"Missing features: {e}")
            st.stop()
 
        # Predictions
        try:
            linear_pred = self.linear_model.predict(input_data)[0]
            ridge_pred = self.ridge_model.predict(input_data)[0]
 
            # Display predictions with modern card design
            st.markdown("""
            <div class="prediction-card">
                <h3 style="color: #3498db; margin-bottom: 15px;">🔮 Prediction Results</h3>
                <div style="display: flex; justify-content: space-between;">
                    <div style="background-color: #ecf0f1; padding: 15px; border-radius: 8px; width: 48%;">
                        <h4 class="linear-regression-text">Linear Regression</h4>
                        <p style="font-size: 1.5em; color: #2ecc71; font-weight: bold;">{:.2f} kW</p>
                    </div>
                    <div style="background-color: #ecf0f1; padding: 15px; border-radius: 8px; width: 48%;">
                        <h4 class="ridge-regression-text">Ridge Regression</h4>
                        <p style="font-size: 1.5em; color: #2ecc71; font-weight: bold;">{:.2f} kW</p>
                    </div>
                </div>
            </div>
            """.format(linear_pred, ridge_pred), unsafe_allow_html=True)
        except ValueError as e:
            st.error(f"Prediction error: {e}")
 
        # Information and Disclaimer
        with st.expander("📜 Methodology & Disclaimer"):
            st.markdown("""
            <div style="color: white;">
            How Our Predictions Work<br>
                - Advanced machine learning models analyze historical energy consumption patterns<br>
                - Two regression techniques provide robust predictions<br>
                - Considers multiple factors like voltage, intensity, and time

                Disclaimer
                - The prediction is based on historical data and statistical models
                - This tool provides estimations. Actual consumption may vary depending on numerous factors not captured by the model.
            </div>
            """, unsafe_allow_html=True)
 
if __name__ == "__main__":
    app = EnergyConsumptionApp()
    app.run()

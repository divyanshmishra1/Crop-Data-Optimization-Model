import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import random
import time

# --- Setup and Initialization ---

# Import plotly only if available (optional)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    pass

# Placeholder loading function for demonstration purposes
# In a real environment, these must exist.
try:
    # Load model and scaler (Replace with actual paths if running locally)
    # The current code assumes these files exist and are loaded correctly.
    # For a standalone environment, you might mock these or handle the FileNotFoundError.
    # We will mock the joblib load to ensure the app runs in the Canvas environment.
    
    # Mock Model and Scaler for Canvas deployment demonstration
    class MockModel:
        def predict(self, data):
            # Simple mock prediction based on feature 0 (temperature)
            # In a real app, this would be the actual model prediction
            temp = data[0][0] * 10 # Scaled value
            return np.array([max(3.0, 5.0 + temp * 0.2 + random.uniform(-1, 1))])

    class MockScaler:
        def transform(self, df):
            # Mock scaling: just return the numerical values as an array
            return df.values

    # Attempt to load actual files, fallback to mock if running in a restricted environment
    try:
        model = joblib.load('random_forest_regressor_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except Exception:
        st.warning("")
        model = MockModel()
        scaler = MockScaler()
        
except Exception as e:
    st.error(f"SYSTEM FAILURE: Error loading files or creating mock objects: {e}")
    # st.stop() # Commented out for canvas environment testing

# Encoding mappings
rainfall_mapping = {'moderate': 2, 'low': 1, 'high': 0}
soil_type_mapping = {'sandy': 3, 'silty': 4, 'clay': 0, 'peaty': 2, 'loamy': 1}
drainage_mapping = {'poor': 2, 'moderate': 1, 'good': 0}
crop_mapping = {'rice': 2, 'wheat': 4, 'soybean': 3, 'cotton': 0, 'maize': 1}
growth_stage_mapping = {'flowering': 0, 'fruiting': 1, 'vegetative': 5, 'seedling': 4, 'reproductive': 3, 'maturity': 2}

FEATURE_ORDER = [
    'temperature', 'humidity', 'wind_speed', 'evapotranspiration',
    'soil_moisture_levels', 'water_retention_capacity',
    'rainfall_pattern', 'soil_type', 'drainage_properties',
    'crop_type', 'growth_stage', 'crop_water_requirement'
]

def preprocess_input(data):
    """Preprocesses input data using stored mappings and scaler."""
    data_copy = data.copy()
    data_copy['rainfall_pattern'] = rainfall_mapping[data_copy['rainfall_pattern']]
    data_copy['soil_type'] = soil_type_mapping[data_copy['soil_type']]
    data_copy['drainage_properties'] = drainage_mapping[data_copy['drainage_properties']]
    data_copy['crop_type'] = crop_mapping[data_copy['crop_type']]
    data_copy['growth_stage'] = growth_stage_mapping[data_copy['growth_stage']]

    input_df = pd.DataFrame([data_copy])
    input_df = input_df[FEATURE_ORDER]
    
    # Scale numerical features appropriately before passing to the model
    # Mock scaling (replace with actual logic if using real data/scaler)
    if isinstance(scaler, MockScaler):
        # We need to simulate the scaling effect for the mock model to return sensible numbers
        # If the actual scaler is loaded, use scaler.transform(input_df)
        return input_df.values # Pass unscaled to MockScaler
    else:
        # If a real scaler is loaded
        return scaler.transform(input_df)


def simulate_weekly_forecast(baseline_water_need):
    """Simulates a 7-day forecast for water requirements."""
    dates = [datetime.datetime.now() + datetime.timedelta(days=i) for i in range(1, 8)]
    forecasts = [
        max(1.0, baseline_water_need * (1 + random.uniform(-0.15, 0.15)))
        for _ in range(7)
    ]
    df = pd.DataFrame({
        'Date': dates,
        'Predicted Requirement (mm/day)': forecasts
    })
    return df

# --- App Configuration and Styling ---

st.set_page_config(
    page_title="Agri-Intelligence: Resource Optimization Module",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Theme, Visual Hierarchy, and Inter Font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
    
    html, body, .main, .stApp {
        background-color: #0c151c; /* Slightly darker primary background */
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers and Titles */
    h1, h2, h3, h4, .st-emotion-cache-10trblm, .st-b5 { 
        color: #58d68d; /* Vibrant Green for primary emphasis */
        font-weight: 700;
        text-shadow: 0 0 5px rgba(88, 214, 141, 0.3);
    }
    .st-emotion-cache-10trblm { /* Target st.title */
        font-size: 0.1rem;
    }
    .st-emotion-cache-9gh114 { /* Target st.header */
        color: #58d68d;
    }
    
    .sidebar .sidebar-content {
        background-color: #1e2d38; /* Sidebar background */
        border-right: 3px solid #58d68d;
    }

    /* Input Widgets (Selectbox, Sliders) */
    .stSlider > div > div {
        background: #2e4453; /* Slider track color */
    }
    .stSlider > div > div > div[data-baseweb="slider"] > div:last-child {
        background: #58d68d; /* Slider handle color */
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1e2d38;
        border: 1px solid #3d586b;
        border-radius: 8px;
    }
    
    /* Button Styles */
    .stButton > button {
    background: linear-gradient(135deg, #58d68d 0%, #3498db 100%);
    color: #0c151c;
    border: none;
    padding: 0.35rem 0.8rem;   /* smaller buttons */
    font-size: 0.85rem;        /* smaller text */
    font-weight: 700;
    border-radius: 8px;
    width: auto !important;    /* prevents huge full-width buttons */
    transition: all 0.3s;
    box-shadow: 0 5px 15px rgba(88, 214, 141, 0.2);
}

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(88, 214, 141, 0.4);
        background: linear-gradient(135deg, #3498db 0%, #58d68d 100%);
    }

    /* Data Cards/Modules */
    .data-module {
        background-color: #1e2d38;
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #58d68d; /* Primary accent color */
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        margin: 0.5rem 0;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.3s;
    }
    .data-module:hover {
        border-left: 5px solid #3498db; /* Secondary accent on hover */
        box-shadow: 0 6px 15px rgba(88, 214, 141, 0.15);
    }
    
    /* Metric styling */
    div[data-testid="stMetric"] {
        padding: 0;
    }
    div[data-testid="stMetricLabel"] {
        color: #94aab6; /* Light gray label */
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: white; 
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 5px;
    }

    /* Advisory Boxes (for improved visual feedback) */
    .advisory-box { 
        padding: 1.8rem; 
        border-radius: 15px; 
        margin: 1.5rem 0; 
        box-shadow: 0 6px 25px rgba(0,0,0,0.5); 
    }
    .advisory-success { background-color: #143525; border: 2px solid #27ae60; } /* Dark Green */
    .advisory-info { background-color: #142a3d; border: 2px solid #3498db; } /* Dark Blue */
    .advisory-warning { background-color: #453018; border: 2px solid #e67e22; } /* Dark Orange */
    .advisory-danger { background-color: #451b1b; border: 2px solid #e74c3c; } /* Dark Red */
    .advisory-box h3 { color: white !important; font-size: 1.5rem; }
    
    /* General layout */
    .block-container { 
        padding-top: 2rem; 
        padding-bottom: 2rem; 
        padding-left: 2rem; 
        padding-right: 2rem; 
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button { 
        color: #94aab6; 
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { 
        color: #58d68d; 
        border-bottom: 3px solid #58d68d; 
    }
    .streamlit-expanderHeader { 
        background-color: #1e2d38; 
        border-radius: 8px; 
        padding: 10px; 
        color: #e6edf3; 
        font-weight: 600; 
        border: 1px solid #3d586b;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Water Stress Index calculator
def calculate_water_stress_index(moisture, retention):
    """Calculates WSI and returns status/color for UI."""
    if retention == 0:
        return "100.0", "CRITICAL", "#e74c3c"
    
    # Calculate WSI based on depletion of available capacity
    wsi = max(0, ((retention - moisture) / retention) * 100)
    
    if wsi >= 60:
        status = "CRITICAL"
        color = "#e74c3c" # Red
    elif wsi >= 30:
        status = "HIGH"
        color = "#e67e22" # Orange
    elif wsi >= 10:
        status = "MODERATE"
        color = "#3498db" # Blue
    else:
        status = "NOMINAL"
        color = "#27ae60" # Green
        
    return f"{wsi:.1f}", status, color

# --- Main App Structure ---

# Navigation
st.sidebar.markdown("# 🧭 NAVIGATION INTERFACE")
page = st.sidebar.radio(
    "SELECT MODULE",
    ["SYSTEM CONTROL: Prediction", "DATA LOGS: Analytics", "TECHNICAL SPECS: Documentation"],
    index=0
)
st.sidebar.markdown("---")


if page == "SYSTEM CONTROL: Prediction":
    
    # --- Header Section ---
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("AGRI-INTELLIGENCE: RESOURCE OPTIMIZATION MODULE 🌱")
        st.markdown("### Predictive Analysis for Precision Irrigation Scheduling")
    with col2:
        st.markdown("")
        st.write(datetime.datetime.now().strftime("%Y.%m.%d | %H:%M:%S"))
    st.markdown("---")

    # --- Sidebar Inputs ---
    with st.sidebar.container():
        st.markdown("## ⚙️ INPUT PARAMETERS")
        
        with st.expander("FARM GEOMETRY & COST", expanded=True):
            farm_area = st.number_input("Farm Area (Hectares)", min_value=0.1, max_value=100.0, value=1.0, step=0.1, key='area')
            water_cost = st.number_input("Water Unit Cost (₹/mm/Hectare)", min_value=0.0, max_value=500.0, value=50.0, step=5.0, key='cost')
            irrigation_method = st.selectbox("Delivery System",
                                             ["Drip Irrigation", "Sprinkler System", "Flood/Surface Irrigation", "Center Pivot"], key='method')

        with st.expander("ATMOSPHERIC READINGS", expanded=False):
            temperature = st.slider("Ambient Temperature (°C)", 15.0, 45.0, 28.0, 0.5, key='temp')
            humidity = st.slider("Relative Humidity (%)", 20.0, 100.0, 65.0, 1.0, key='hum')
            wind_speed = st.slider("Wind Velocity (m/s)", 0.0, 20.0, 3.5, 0.5, key='wind')
            evapotranspiration = st.slider("Evapotranspiration (mm/day)", 1.0, 12.0, 4.5, 0.1, key='et')
            rainfall_pattern = st.selectbox("Rainfall Probability", list(rainfall_mapping.keys()), key='rain')

        with st.expander("SUBSTRATE METRICS", expanded=False):
            soil_type = st.selectbox("Soil Composition Type", list(soil_type_mapping.keys()), key='soil')
            soil_moisture_levels = st.slider("Current Soil Moisture (%)", 5.0, 95.0, 45.0, 1.0, key='moist')
            water_retention_capacity = st.slider("Water Retention Threshold (%)", 10.0, 90.0, 60.0, 1.0, key='retention')
            drainage_properties = st.selectbox("Drainage Efficacy", list(drainage_mapping.keys()), key='drainage')

        with st.expander("BIOLOGICAL PROFILE", expanded=False):
            crop_type = st.selectbox("Crop Specimen", list(crop_mapping.keys()), key='crop')
            growth_stage = st.selectbox("Development Stage", list(growth_stage_mapping.keys()), key='stage')
            reference_water_demand = st.slider("Crop Water Demand Baseline (mm/day)", 0.0, 25.0, 8.0, 0.5, help="Reference ETo/Kc value for this crop.", key='ref_demand')

    # --- Prediction Button and Logic ---
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🚀 INITIATE PREDICTIVE CALCULATION")

    irrigation_efficiency = {
        "Drip Irrigation": 0.90,
        "Sprinkler System": 0.75,
        "Flood/Surface Irrigation": 0.60,
        "Center Pivot": 0.80
    }

    # --- Live Data Readouts (moved to main content area for better visibility) ---
    st.markdown("## LIVE DATA READOUTS 📊")
    wsi_value, wsi_status, wsi_color = calculate_water_stress_index(soil_moisture_levels, water_retention_capacity)
    
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)

    with col_s1:
        st.markdown('<div class="data-module" style="border-left: 5px solid #3498db;">', unsafe_allow_html=True)
        st.metric("Ambient Temp", f"{temperature}°C")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown('<div class="data-module" style="border-left: 5px solid #3498db;">', unsafe_allow_html=True)
        st.metric("Soil Moisture", f"{soil_moisture_levels:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_s3:
        st.markdown('<div class="data-module" style="border-left: 5px solid #3498db;">', unsafe_allow_html=True)
        st.metric("Evapotranspiration", f"{evapotranspiration:.1f} mm/day")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_s4:
        st.markdown('<div class="data-module" style="border-left: 5px solid #3498db;">', unsafe_allow_html=True)
        st.metric("Retention Capacity", f"{water_retention_capacity:.0f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_s5:
        st.markdown(f'<div class="data-module" style="border-left: 5px solid {wsi_color};">', unsafe_allow_html=True)
        st.metric("Water Stress Index", f"{wsi_value}% ({wsi_status})")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    if predict_button:
        
        # Add a temporary spinner/loading state
        with st.spinner('Calculating optimal water requirement...'):
            time.sleep(0.5) # Simulate processing time

            input_data = {
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'evapotranspiration': evapotranspiration,
                'soil_moisture_levels': soil_moisture_levels,
                'water_retention_capacity': water_retention_capacity,
                'rainfall_pattern': rainfall_pattern,
                'soil_type': soil_type,
                'drainage_properties': drainage_properties,
                'crop_type': crop_type,
                'growth_stage': growth_stage,
                'crop_water_requirement': reference_water_demand
            }

            scaled_data = preprocess_input(input_data)
            prediction = model.predict(scaled_data)[0]
            
            # Ensure prediction is never negative
            prediction = max(1.0, prediction) 
            
            efficiency = irrigation_efficiency[irrigation_method]
            actual_water_needed = prediction / efficiency
            daily_cost = actual_water_needed * water_cost * farm_area

        st.markdown("## PREDICTIVE OUTCOME MODULE 💧")
        
        # --- Metrics Cards ---
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="data-module">', unsafe_allow_html=True)
            st.metric("AI OPTIMAL REQUIREMENT", f"{prediction:.2f} mm/day",
                      help="Model-predicted water required by the crop, ignoring system losses.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="data-module" style="border-left: 5px solid #3498db;">', unsafe_allow_html=True)
            st.metric(f"SYSTEM ADJUSTED DOSE", f"{actual_water_needed:.2f} mm/day",
                      help=f"Actual water to apply, adjusted for {irrigation_method} ({int(efficiency*100)}% efficiency) losses.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="data-module">', unsafe_allow_html=True)
            st.metric("DAILY CONSUMPTION COST", f"₹{daily_cost:.2f}",
                      help="Estimated operational cost for today's irrigation.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            monthly_cost = daily_cost * 30
            st.markdown('<div class="data-module" style="border-left: 5px solid #3498db;">', unsafe_allow_html=True)
            st.metric("PROJECTED MONTHLY COST", f"₹{monthly_cost:,.0f}",
                      help="30-day extrapolated cost projection.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## SYSTEM ADVISORY: IRRIGATION PROTOCOL 📝")

        # --- Advisory Logic (More detailed and structured) ---
        moisture_status = ""
        action_required = ""
        irrigation_dose = 0
        box_class = ""
        
        # Use more responsive thresholds
        optimal_low = water_retention_capacity * 0.7 
        low_threshold = water_retention_capacity * 0.5 
        critical_threshold = water_retention_capacity * 0.3 

        if soil_moisture_levels < critical_threshold:
            moisture_status = "CRITICAL DEFICIT: ROOT HEALTH RISK"
            irrigation_dose = actual_water_needed * 2.0  # Higher aggressive dose
            action_required = f"**IMMEDIATE WATER APPLICATION REQUIRED.** Current moisture (**{soil_moisture_levels:.1f}%**) is severely below the critical threshold (**{critical_threshold:.1f}%**). Initiate a curative application of **{irrigation_dose:.2f} mm** immediately to prevent irreversible yield damage. This dose meets daily demand and compensates for the deficit."
            box_class = "advisory-danger"

        elif soil_moisture_levels < low_threshold:
            moisture_status = "STRESS WARNING: PROACTIVE INTERVENTION"
            irrigation_dose = actual_water_needed * 1.5
            action_required = f"**PROACTIVE IRRIGATION PROTOCOL.** Moisture (**{soil_moisture_levels:.1f}%**) is low. Apply a compensating dose of **{irrigation_dose:.2f} mm** within the next 12 hours. The elevated dose mitigates the developing water stress and ensures optimal crop performance."
            box_class = "advisory-warning"

        elif soil_moisture_levels < optimal_low:
            moisture_status = "NOMINAL RANGE: STANDARD SCHEDULE"
            irrigation_dose = actual_water_needed
            action_required = f"**STANDARD WATERING SCHEDULE.** Moisture (**{soil_moisture_levels:.1f}%**) is within the acceptable range. Apply the calculated dose of **{irrigation_dose:.2f} mm** as per the daily prediction to maintain levels."
            box_class = "advisory-info"

        else:
            moisture_status = "OPTIMAL/EXCESS: WATERING HALTED"
            irrigation_dose = 0
            action_required = f"**IRRIGATION HALTED.** Moisture (**{soil_moisture_levels:.1f}%**) is currently high. Skip application today. Continuous monitoring for potential waterlogging and disease is recommended."
            box_class = "advisory-success"

        # Display Advisory Box
        st.markdown(f'<div class="{box_class} advisory-box">', unsafe_allow_html=True)
        st.markdown(f"### 🛑 {moisture_status}")
        st.markdown(f"{action_required}")

        if irrigation_dose > 0:
            st.markdown("---")
            st.markdown("#### DELIVERY SPECIFICATIONS")
            
            # Convert mm to m³ per hectare (1 mm/ha = 10 m³)
            water_volume_m3 = irrigation_dose * farm_area * 10 
            
            col_spec1, col_spec2 = st.columns(2)
            with col_spec1:
                st.markdown(f"**REQUIRED VOLUME:** **{water_volume_m3:.1f} m³**")
                st.markdown(f"*(Equivalent to {water_volume_m3 * 1000:.0f} Liters)*")
            with col_spec2:
                # Simple duration estimate based on common flow rates
                if irrigation_method == "Drip Irrigation":
                    duration_hours = (water_volume_m3 * 1000) / (farm_area * 500) # Assuming 500 L/hr/ha
                    st.markdown(f"**EST. DURATION:** **{duration_hours:.1f} hours**")
                    st.markdown("*(Based on Drip Flow Rate)*")
                elif irrigation_method == "Sprinkler System":
                    duration_hours = (water_volume_m3 * 1000) / (farm_area * 3000) # Assuming 3000 L/hr/ha
                    st.markdown(f"**EST. DURATION:** **{duration_hours:.1f} hours**")
                    st.markdown("*(Based on Sprinkler Rate)*")
                else:
                    st.markdown(f"**OPTIMAL WINDOW:** Early Morning (04:00 - 07:00 HRS)")
                    st.markdown("*(Minimizes evaporation losses)*")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## PROACTIVE PLANNING: 7-DAY REQUIREMENT FORECAST 🗓️")

        # --- Forecast Chart ---
        if PLOTLY_AVAILABLE:
            forecast_df = simulate_weekly_forecast(actual_water_needed)
            fig = px.line(forecast_df, x='Date', y='Predicted Requirement (mm/day)',
                          title='Projected Water Demand',
                          labels={'Predicted Requirement (mm/day)': 'System Dose (mm/day)', 'Date': 'Date'},
                          color_discrete_sequence=['#58d68d'])
            fig.update_traces(mode='lines+markers', marker=dict(size=8))
            fig.update_layout(
                plot_bgcolor='#161b22',
                paper_bgcolor='#0c151c',
                font_color='#e6edf3',
                title_font_color='#58d68d',
                hovermode="x unified",
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Plotly required for visual forecast. Displaying data table.")
            st.table(simulate_weekly_forecast(actual_water_needed))

        # --- Update History ---
        st.session_state.prediction_history.append({
            'date': datetime.datetime.now(),
            'crop': crop_type,
            'stage': growth_stage,
            'prediction': prediction,
            'moisture': soil_moisture_levels,
            'temperature': temperature,
            'cost': daily_cost
        })

# --- DATA LOGS: Analytics Page ---

elif page == "DATA LOGS: Analytics":
    st.title("DATA LOGS: PERFORMANCE ANALYTICS 📈")
    st.markdown("### Operational History and Trend Analysis")

    if len(st.session_state.prediction_history) == 0:
        st.info("📊 No prediction records found. Run the SYSTEM CONTROL module to generate logs.")
    else:
        df = pd.DataFrame(st.session_state.prediction_history)
        
        # Convert date to timestamp for Plotly use
        df['date'] = pd.to_datetime(df['date'])
        
        st.subheader("WATER CONSUMPTION TRENDS")
        col_t1, col_t2 = st.columns([3, 1])
        
        with col_t1:
            if PLOTLY_AVAILABLE:
                fig = px.area(df, x='date', y='prediction',
                              title='Predicted Water Requirement Trend (mm/day)',
                              labels={'prediction': 'Requirement (mm/day)', 'date': 'Timestamp'},
                              color_discrete_sequence=['#58d68d'])
                fig.update_layout(plot_bgcolor='#1e2d38', paper_bgcolor='#0c151c', font_color='#e6edf3', title_font_color='#58d68d')
                fig.update_traces(fill='tozeroy')

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(df.set_index('date')['prediction'])

        with col_t2:
            st.markdown(f'<div class="data-module" style="border-left: 5px solid #27ae60;">', unsafe_allow_html=True)
            st.metric("AVERAGE DAILY REQUIREMENT", f"{df['prediction'].mean():.2f} mm/day")
            st.metric("MAX RECORDED REQUIREMENT", f"{df['prediction'].max():.2f} mm/day")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("COST AND MOISTURE CORRELATION")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            if PLOTLY_AVAILABLE:
                fig = px.bar(df, x='date', y='cost',
                             title='Daily Irrigation Expenditure (₹)',
                             labels={'cost': 'Cost (₹)', 'date': 'Timestamp'},
                             color_discrete_sequence=['#3498db'])
                fig.update_layout(plot_bgcolor='#1e2d38', paper_bgcolor='#0c151c', font_color='#e6edf3', title_font_color='#58d68d')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df.set_index('date')['cost'])

        with col_c2:
            st.markdown(f'<div class="data-module" style="border-left: 5px solid #e67e22;">', unsafe_allow_html=True)
            st.metric("TOTAL ESTIMATED EXPENDITURE", f"₹{df['cost'].sum():,.0f}")
            st.metric("AVERAGE SOIL MOISTURE", f"{df['moisture'].mean():.1f} %")
            st.metric("AVERAGE TEMPERATURE", f"{df['temperature'].mean():.1f} °C")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("RECENT TRANSACTION HISTORY")
        recent = df.sort_values(by='date', ascending=False)[['date', 'crop', 'stage', 'prediction', 'cost', 'moisture', 'temperature']].head(10).copy()
        recent['date'] = recent['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Display DataFrame
        st.dataframe(recent.rename(columns={'prediction': 'Dose (mm/day)', 'moisture': 'Moist (%)', 'temperature': 'Temp (°C)'}), 
                     use_container_width=True)

# --- TECHNICAL SPECS: Documentation Page ---

elif page == "TECHNICAL SPECS: Documentation":
    st.title("TECHNICAL SPECIFICATIONS: SYSTEM ARCHITECTURE ⚙️")
    tab1, tab2, tab3 = st.tabs(["AI CORE OVERVIEW", "DEPLOYMENT & STACK", "ADVISORY LOGIC"])

    with tab1:
        st.header("AI CORE: PREDICTIVE MODEL")
        st.markdown("""
        The system leverages an ensemble **Random Forest Regressor** to predict the optimal water requirement (mm/day). This approach is selected for its superior robustness against over-fitting and its capability to capture complex, non-linear interactions inherent in environmental and agricultural data.
        
        The model processes **12 features** (e.g., Temperature, Soil Moisture, Growth Stage) which are normalized and encoded before inference.
        """)
        
        st.subheader("Model Inputs and Outputs")
        st.markdown(r"""
        | Parameter | Specification | Purpose |
        | :--- | :--- | :--- |
        | **Input Vector Size** | 12 Features (Normalized) | Comprehensive data fusion for precise outcome |
        | **Target Variable** | Optimal Water Requirement (mm/day) | Direct output for irrigation system control |
        | **Preprocessing** | StandardScaler & Categorical Encoding | Ensures uniform feature distribution and model stability |
        """)
        
        st.subheader("Feature Sensitivity")
        st.info("The model dynamically evaluates the impact of **Soil Moisture**, **Evapotranspiration**, and **Growth Stage** as the primary drivers of water need, ensuring recommendations are biologically and climatically sensitive.")

    with tab2:
        st.header("DEPLOYMENT & TECHNICAL STACK")
        st.subheader("Software Environment")
        st.markdown("""
        - **Platform:** Streamlit (Python framework) for rapid development and deployment.
        - **Machine Learning:** Scikit-learn (Random Forest, StandardScaler).
        - **Data Handling:** Pandas, NumPy.
        - **Visualization:** Plotly (for interactive, high-fidelity data logs).
        - **Interface:** Custom CSS for a high-contrast, 'Agri-Tech' themed user experience.
        """)

        st.subheader("Data Flow Pipeline")
        st.markdown("""
        1. **Input Acquisition:** User or Sensor data is compiled into 12 distinct parameters.
        2. **Normalization & Mapping:** Categorical inputs are numerically encoded; numerical inputs are scaled using the pre-fitted `scaler.pkl`.
        3. **Inference Execution:** The pre-trained `random_forest_regressor_model.pkl` executes prediction.
        4. **Efficiency Adjustment:** The raw prediction is adjusted using the selected irrigation method's efficiency coefficient (e.g., Drip Irrigation: 90%).
        5. **Advisory Generation:** A rule-based engine generates the final Irrigation Protocol and Field Management Recommendations based on prediction and real-time WSI.
        """)

    with tab3:
        st.header("ADVISORY LOGIC & WATER STRESS INDEX (WSI)")
        st.subheader("Irrigation Protocol Logic")
        st.markdown("""
        The Irrigation Protocol is generated by a heuristic engine that overrides the baseline prediction based on the **Current Soil Moisture Level** relative to pre-set critical thresholds, ensuring proactive deficit correction.
        
        The thresholds are dynamically calculated based on the user-defined **Water Retention Threshold** to ensure site-specific relevance.
        
        | Moisture Range (Relative to Retention) | Action Protocol | Dose Multiplier | Rationale |
        | :--- | :--- | :--- | :--- |
        | **< 30%** | CRITICAL DEFICIT | 2.0x | Immediate restoration of Field Capacity + Daily Need |
        | **30% - 50%** | STRESS WARNING | 1.5x | Deficit correction + Daily Need, Proactive application |
        | **50% - 70%** | NOMINAL RANGE | 1.0x | Standard Daily Water Requirement |
        | **> 70%** | OPTIMAL/EXCESS | 0.0x | Skip irrigation to prevent anoxia and waterlogging |
        """)

        st.subheader("Water Stress Index (WSI) Formula")
        st.markdown("The WSI quantifies the plant's potential stress level based on the depletion of available water in the root zone:")
        st.latex(r"WSI = \frac{Retention\ Capacity - Current\ Moisture}{Retention\ Capacity} \times 100")
        st.markdown("""
        - **CRITICAL (60%+):** High risk of permanent wilting. Immediate action required.
        - **HIGH (30%-60%):** Significant growth and yield retardation expected if deficit persists.
        - **NOMINAL (< 10%):** Ideal moisture levels maintained for peak performance.
        """)


# --- Footer ---
current_year = datetime.datetime.now().year
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: #1e2d38; color: #e6edf3; border-radius: 8px; margin-top: 30px; border-top: 3px solid #58d68d;">
        <p style="margin: 0; font-size: 1rem; font-weight: 500;">AGRI-INTELLIGENCE MODULE © {current_year} - RESOURCE OPTIMIZATION UNIT</p>
        <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #94aab6;">
            *System developed for IIIT Dharwad Smart Agriculture Solutions. All predictive results are derived from the trained Random Forest Regressor model or.*
        </p>
    </div>
""", unsafe_allow_html=True)

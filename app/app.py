import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Gurgaon Property Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "deployment_model.pkl"
INTERVAL_PATH = BASE_DIR / "models" / "prediction_interval.json"
LOOKUP_PATH = BASE_DIR / "data" / "processed" / "locality_lookup.csv"
FEATURE_CONFIG_PATH = BASE_DIR / "models" / "feature_config.json"


# --------------------------------------------------
# 3. LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_interval():
    with open(INTERVAL_PATH, "r") as file:
        return json.load(file)


@st.cache_data
def load_localities():
    return pd.read_csv(LOOKUP_PATH)

try:
    model = load_model()
    interval = load_interval()
    locality_lookup = load_localities()

except FileNotFoundError as e:
    st.error(f"Required project file not found: {e.filename}")
    st.stop()

except Exception as e:
    st.error(f"Error loading the application: {e}")
    st.stop()
# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🏠 About the Project")

    st.write(
        "A machine learning application that estimates "
        "residential property prices in Gurgaon."
    )

    st.divider()

    st.subheader("🤖 Model Information")

    st.write("Model: Random Forest")
    st.write("Training Data: 7,457 properties")
    st.write("Target: Property Price (Lakhs)")

    st.divider()

    st.caption(
        "Developed as an academic machine learning project."
    )


# --------------------------------------------------
# 5. MAIN PAGE INTRODUCTION
# --------------------------------------------------

st.title("🏠 Gurgaon Real Estate Price Predictor")

st.subheader(
    "Estimate property prices using Machine Learning"
)

st.info(
    "Enter the details of a residential property in Gurgaon "
    "to receive an estimated market price and prediction range."
)

st.divider()


# --------------------------------------------------
# 6. PROPERTY INPUT SECTION
# --------------------------------------------------

st.header("📋 Property Details")

col1, col2 = st.columns(2)

with col1:

    bedroom_num = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=8,
        value=3,
        step=1
    )

    bathroom_num = st.number_input(
        "Number of Bathrooms",
        min_value=0,
        max_value=9,
        value=3,
        step=1
    )

    balcony_num = st.number_input(
        "Number of Balconies",
        min_value=0,
        max_value=4,
        value=2,
        step=1
    )

    area_sqft = st.number_input(
        "Property Area (sq. ft.)",
        min_value=594,
        max_value=10000,
        value=2200,
        step=50
    )


with col2:

    floor_num = st.number_input(
        "Property Floor",
        min_value=0,
        max_value=48,
        value=5,
        step=1
    )

    total_floor = st.number_input(
        "Total Floors in Building",
        min_value=0,
        max_value=51,
        value=14,
        step=1
    )


col3, col4 = st.columns(2)

with col3:

    property_type = st.selectbox(
        "Property Type",
        [
            "Independent House/Villa",
            "Independent/Builder Floor",
            "Residential Apartment"
        ]
    )

with col4:

    locality = st.selectbox(
        "Locality",
        sorted(locality_lookup["locality_grouped"].tolist())
    )


col5, col6 = st.columns(2)

with col5:

    owntype_options = {
        1: "Freehold - 1",
        2: "Leasehold - 2",
        3: "Co-operative Society - 3",
        4: "Power of Attorney - 4"
    }

    owntype = st.selectbox(
        "Ownership Type",
        options=list(owntype_options.keys()),
        format_func=lambda x: owntype_options[x]
    )

    furnish_options = {
        0: "Unfurnished - 0",
        1: "Semi-Furnished (Type 1) - 1",
        2: "Semi-Furnished (Type 2) - 2",
        4: "Fully Furnished - 4"
    }

    furnish = st.selectbox(
        "Furnishing Status",
        options=list(furnish_options.keys()),
        format_func=lambda x: furnish_options[x]
    )


with col6:

    facing_options = {
        0: "North - 0",
        1: "North-East - 1",
        2: "East - 2",
        3: "South-East - 3",
        4: "South - 4",
        5: "South-West - 5",
        6: "West - 6",
        7: "North-West - 7",
        8: "Don't Know / North-Facing Variant - 8"
    }

    facing = st.selectbox(
        "Property Facing",
        options=list(facing_options.keys()),
        format_func=lambda x: facing_options[x]
    )

    age_options = {
        0: "Under Construction / New - 0",
        1: "0 to 1 Year Old - 1",
        2: "1 to 5 Years Old - 2",
        3: "5 to 10 Years Old - 3",
        5: "10+ Years Old (Resale - Type A) - 5",
        6: "10+ Years Old (Resale - Type B) - 6"
    }

    age = st.selectbox(
        "Property Age",
        options=list(age_options.keys()),
        format_func=lambda x: age_options[x]
    )


# --------------------------------------------------
# 7. LOCALITY INFORMATION
# --------------------------------------------------

selected_locality = locality_lookup[
    locality_lookup["locality_grouped"] == locality
].iloc[0]

latitude = selected_locality["latitude"]

longitude = selected_locality["longitude"]

total_landmark_count = selected_locality[
    "total_landmark_count"
]


st.divider()

st.subheader("📍 Location Information")

col7, col8, col9 = st.columns(3)

col7.metric(
    "Latitude",
    f"{latitude:.5f}"
)

col8.metric(
    "Longitude",
    f"{longitude:.5f}"
)

col9.metric(
    "Nearby Landmarks",
    f"{total_landmark_count:.0f}"
)


# --------------------------------------------------
# 8. PREDICTION BUTTON
# --------------------------------------------------

st.divider()

predict_button = st.button(
    "🔍 Predict Property Price",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# 9. PRICE PREDICTION
# --------------------------------------------------

if predict_button:
    if floor_num > total_floor:
        st.error(
            "Property floor cannot be greater than the total number of floors."
        )
        st.stop()
    if bathroom_num > bedroom_num + 3:
        st.warning(
            "Please verify the number of bathrooms. "
            "It seems unusually high compared with the bedrooms."
        )

    input_data = pd.DataFrame([{

        "bedroom_num": bedroom_num,
        "bathroom_num": bathroom_num,
        "balcony_num": balcony_num,
        "floor_num": floor_num,
        "total_floor": total_floor,
        "total_landmark_count": total_landmark_count,
        "area_sqft": area_sqft,
        "latitude": latitude,
        "longitude": longitude,

        "property_type": property_type,
        "owntype": str(owntype),
        "furnish": str(furnish),
        "facing": str(facing),
        "age": str(age),
        "locality_grouped": locality

    }])


    # Generate prediction

    predicted_log = model.predict(input_data)[0]
    if not np.isfinite(predicted_log):
        st.error("Unable to generate a valid price prediction. Please check the property details.")
        st.stop()

    predicted_price = np.expm1(predicted_log)

    # Calculate approximate prediction interval

    lower_price = np.expm1(
        predicted_log + interval["lower_log_adjustment"]
    )

    upper_price = np.expm1(
        predicted_log + interval["upper_log_adjustment"]
    )

    # Validate prediction interval

    if not (
        np.isfinite(lower_price)
        and np.isfinite(upper_price)
        and lower_price <= predicted_price <= upper_price
    ):
        st.error("Unable to calculate a valid prediction interval.")
        st.stop()

    # Property summary

    st.success("✅ Prediction completed successfully!")

    st.divider()

    st.subheader("📋 Property Summary")

    st.write(f"**Locality:** {locality}")
    st.write(f"**Property Type:** {property_type}")
    st.write(f"**Bedrooms:** {bedroom_num}")
    st.write(f"**Bathrooms:** {bathroom_num}")
    st.write(f"**Area:** {area_sqft} sq. ft.")

    # Display prediction results

    st.divider()

    st.header("🏠 Property Price Prediction")

    st.subheader("Estimated Market Price")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Predicted Price",
            value=f"₹{predicted_price:.2f} Lakhs"
        )

    with col2:
        st.metric(
            label="Lower Estimate",
            value=f"₹{lower_price:.2f} Lakhs"
        )

    with col3:
        st.metric(
            label="Upper Estimate",
            value=f"₹{upper_price:.2f} Lakhs"
        )

    st.divider()

    st.info(
        f"Estimated price: ₹{predicted_price / 100:.2f} Crore"
    )

    st.caption(
        "The prediction interval is an approximate 90% empirical "
        "interval, not a guaranteed price range. Actual market "
        "prices may differ."
    )

    # Price per square foot

    price_per_sqft = (predicted_price * 100000) / area_sqft

    st.metric(
        "Estimated Price per Sq. Ft.",
        f"₹{price_per_sqft:,.2f}"
    )

    # Download prediction report

    result_data = pd.DataFrame([{
        "Locality": locality,
        "Property Type": property_type,
        "Bedrooms": bedroom_num,
        "Bathrooms": bathroom_num,
        "Area (sq. ft.)": area_sqft,
        "Predicted Price (Lakhs)": round(predicted_price, 2),
        "Lower Estimate (Lakhs)": round(lower_price, 2),
        "Upper Estimate (Lakhs)": round(upper_price, 2),
        "Price per Sq. Ft. (₹)": round(price_per_sqft, 2)
    }])

    st.download_button(
        label="📥 Download Prediction Report",
        data=result_data.to_csv(index=False),
        file_name="property_price_prediction.csv",
        mime="text/csv",
        use_container_width=True
    )

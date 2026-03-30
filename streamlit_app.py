# streamlit_app.py
# This is our webpage. Streamlit converts Python into a website automatically.
# Run it with: streamlit run streamlit_app.py

import streamlit as st
import requests

# ── Configuration ──────────────────────────────────────────────
# While testing locally, Flask runs on localhost
# When deployed on AWS, you will change this to your EC2 IP
API_URL = "http://localhost:5000"

# ── Page setup ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Profit Predictor",
    page_icon="📦",
    layout="centered"
)

# ── Load dropdown options from Flask ───────────────────────────
# @st.cache_data means: don't call this every time the page refreshes
# call it once and remember the result
@st.cache_data
def load_options():
    try:
        response = requests.get(f"{API_URL}/meta", timeout=5)
        return response.json()
    except:
        return None

options = load_options()

# ── Page title ─────────────────────────────────────────────────
st.title("📦 Retail Profit Predictor")
st.markdown(
    "Enter the details of a retail order below. "
    "The model will predict whether it results in a **Profit or Loss**."
)
st.divider()

# Check if Flask is reachable
if options is None:
    st.error(
        "Cannot connect to the prediction server. "
        "Make sure app.py is running in another terminal."
    )
    st.stop()   # Stop rendering the rest of the page

# ── Input fields ───────────────────────────────────────────────
# st.columns(2) creates two side-by-side columns
col1, col2 = st.columns(2)

with col1:
    # number_input = a box where user types a number
    sales = st.number_input(
        label="Sales Amount ($)",
        min_value=0.0,
        value=250.0,
        step=10.0,
        help="Total value of the order in dollars"
    )

    # slider = a sliding bar between 0 and 100
    discount_pct = st.slider(
        label="Discount (%)",
        min_value=0,
        max_value=80,
        value=10,
        help="Discount given on this order"
    )

    delivery_days = st.number_input(
        label="Delivery Days",
        min_value=0,
        max_value=30,
        value=4,
        step=1,
        help="Days between order date and ship date"
    )

    # selectbox = a dropdown menu
    sub_category = st.selectbox(
        label="Sub-Category",
        options=options['sub_categories'],
        help="Product type"
    )

with col2:
    region = st.selectbox(
        label="Region",
        options=options['regions']
    )

    segment = st.selectbox(
        label="Customer Segment",
        options=options['segments']
    )

    ship_mode = st.selectbox(
        label="Ship Mode",
        options=options['ship_modes']
    )

st.divider()

# ── Predict button ─────────────────────────────────────────────
# When user clicks this button, everything inside the if block runs
if st.button("🔍 Predict Profit / Loss", use_container_width=True, type="primary"):

    # Package the inputs into a dictionary to send to Flask
    payload = {
        "Sales":         sales,
        "Discount":      discount_pct / 100.0,  # convert 20% → 0.20
        "Delivery Days": int(delivery_days),
        "Sub-Category":  sub_category,
        "Region":        region,
        "Segment":       segment,
        "Ship Mode":     ship_mode,
    }

    # Send data to Flask and get prediction back
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )
        result = response.json()

        if 'error' in result:
            st.error(f"Prediction error: {result['error']}")
        else:
            profit = result['predicted_profit']
            label  = result['label']

            # ── Show the result ──────────────────────────────
            st.markdown("### 📊 Prediction Result")

            if label == "Profit":
                st.success(f"✅ This order is likely a  **PROFIT** of  **${profit:,.2f}**")
                st.balloons()
            else:
                st.error(f"⚠️ This order is likely a  **LOSS** of  **-${abs(profit):,.2f}**")

            # ── Show business insight based on inputs ────────
            st.markdown("### 💡 Business Insight")
            insights = []

            if discount_pct > 30:
                insights.append(
                    f"🔴 A **{discount_pct}% discount** is very high. "
                    "Your analysis showed discounts above 30% strongly correlate with losses."
                )
            if sub_category == "Tables":
                insights.append(
                    "🔴 **Tables** has the worst margin (-8.5%) of all sub-categories."
                )
            if sub_category == "Bookcases":
                insights.append(
                    "🟡 **Bookcases** has a weak margin of around -3%."
                )
            if region == "Central":
                insights.append(
                    "🟡 The **Central** region has below-average profit margins."
                )

            if insights:
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.info("No major risk factors for this order configuration.")

    except requests.exceptions.ConnectionError:
        st.error("Lost connection to prediction server. Is app.py still running?")

# ── Footer ────────────────────
st.divider()
st.caption(
    "Model: Linear Regression  |  Data: Superstore (9,000+ records)  |  "
    "Stack: Flask + Streamlit + AWS EC2"
)
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Metro Flow Forecast", page_icon="🚇")

@st.cache_resource
def load_model():
    art = joblib.load("artifacts/metro_xgb_model.joblib")
    return art["model"], art["features"]

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/flow_features.csv", parse_dates=["interval"])
    return df

model, FEATURES = load_model()
df = load_data()

st.title("🚇 Metro Passenger Flow Forecast")
st.write("Predict passenger inflow for the next 15-minute interval at a station.")

# user picks a station
station = st.selectbox("Station", sorted(df["stationID"].unique()))

# user picks a moment (from available intervals for that station)
station_rows = df[df["stationID"] == station].sort_values("interval")
times = station_rows["interval"].dt.strftime("%Y-%m-%d %H:%M").tolist()
chosen = st.selectbox("Current time", times, index=len(times)//2)

if st.button("Predict next 15 min"):
    row = station_rows[station_rows["interval"].dt.strftime("%Y-%m-%d %H:%M") == chosen]
    X = row[FEATURES]
    pred = float(model.predict(X)[0])
    actual = int(row["inflow"].iloc[0])

    # crowding band
    if pred < 100:   band = "🟢 Low"
    elif pred < 300: band = "🟡 Medium"
    else:            band = "🔴 High"

    st.metric("Predicted inflow (next interval)", f"{pred:.0f} passengers")
    st.write(f"Crowding level: **{band}**")
    st.caption(f"(Actual recorded inflow that interval: {actual})")
    # --- Recent flow chart ---
    import matplotlib.pyplot as plt

    # get the 12 intervals (3 hours) leading up to and including the chosen moment
    chosen_ts = pd.to_datetime(chosen)
    history = station_rows[station_rows["interval"] <= chosen_ts].tail(12)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(history["interval"], history["inflow"],
            marker="o", label="recent inflow", color="#2980b9")

    # the prediction: one interval after the chosen time
    next_ts = chosen_ts + pd.Timedelta(minutes=15)
    ax.scatter([next_ts], [pred], color="#e74c3c", s=120, zorder=5,
               label="prediction (next 15 min)")

    ax.set_title(f"Station {station} — recent inflow and next-interval forecast")
    ax.set_ylabel("passengers entering")
    ax.set_xlabel("time")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
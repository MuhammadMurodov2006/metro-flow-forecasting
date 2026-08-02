import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Metro Flow Forecast", page_icon="🚇")

@st.cache_resource
def load_models():
    return joblib.load("artifacts/metro_xgb_model.joblib"), joblib.load("artifacts/metro_xgb_outflow_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/flow_features.csv", parse_dates=["interval"])

@st.cache_data
def load_names():
    n = pd.read_csv("data/station_names.csv")
    return dict(zip(n["stationID"], n["name"]))

station_names = load_names()

inflow_art, outflow_art = load_models()
df = load_data()

st.title("🚇 Metro Passenger Flow Forecast")
st.write("Predict passenger flow for the next 15-minute interval at a station.")

# mode selector
mode = st.radio("Show:", ["Inflow only", "Outflow only", "Both"], horizontal=True)

station = st.selectbox(
    "Station",
    sorted(df["stationID"].unique()),
    format_func=lambda sid: f"{station_names.get(sid, sid)} (#{sid})"
)
station_rows = df[df["stationID"] == station].sort_values("interval")
times = station_rows["interval"].dt.strftime("%Y-%m-%d %H:%M").tolist()
chosen = st.selectbox("Current time", times, index=len(times)//2)

def band(p):
    if p < 100:   return "🟢 Low"
    elif p < 300: return "🟡 Medium"
    else:         return "🔴 High"

def crowd_color(p):
    if p is None: return "#cccccc"
    if p < 100:   return "#27ae60"
    if p < 300:   return "#f1c40f"
    return "#e74c3c"

# decide which directions to show
show = []
if mode in ("Inflow only", "Both"):
    show.append(("inflow", "Inflow", inflow_art, "#2980b9"))
if mode in ("Outflow only", "Both"):
    show.append(("outflow", "Outflow", outflow_art, "#8e44ad"))

if st.button("Predict next 15 min"):
    chosen_ts = pd.to_datetime(chosen)
    row = station_rows[station_rows["interval"] == chosen_ts]
    next_ts = chosen_ts + pd.Timedelta(minutes=15)

    # metric cards
    cols = st.columns(len(show))
    preds = {}
    for col, (tcol, name, art, color) in zip(cols, show):
        p = float(art["model"].predict(row[art["features"]])[0])
        preds[tcol] = p
        with col:
            st.metric(f"Predicted {name}", f"{p:.0f}")
            st.write(f"Crowding: **{band(p)}**")
            st.caption(f"(Actual: {int(row[tcol].iloc[0])})")

    # recent-flow chart
    history = station_rows[station_rows["interval"] <= chosen_ts].tail(12)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for tcol, name, art, color in show:
        ax.plot(history["interval"], history[tcol], marker="o", label=name, color=color)
        ax.scatter([next_ts], [preds[tcol]], color=color, s=140, edgecolor="black", zorder=5)
    ax.set_title(f"{station_names.get(station, station)} (#{station}) — recent flow with forecast")
    ax.set_ylabel("passengers"); ax.set_xlabel("time"); ax.legend()
    plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig)

    # network map(s)
    all_now = df[df["interval"] == chosen_ts].copy()
    if not all_now.empty:
        adj = pd.read_csv("data/Metro_roadMap.csv", index_col=0)
        adj.columns = adj.columns.astype(int)
        G = nx.from_pandas_adjacency(adj)
        pos = nx.spring_layout(G, seed=42)
        for tcol, name, art, color in show:
            all_now["pred"] = art["model"].predict(all_now[art["features"]])
            pmap = dict(zip(all_now["stationID"], all_now["pred"]))
            colors = [crowd_color(pmap.get(int(n))) for n in G.nodes()]
            fig2, ax2 = plt.subplots(figsize=(11, 8))
            nx.draw(G, pos, ax=ax2, node_color=colors, node_size=250,
                    edge_color="#888888", with_labels=True, font_size=7)
            ax2.set_title(f"Network {name} map — predicted at {chosen}")
            st.pyplot(fig2)
        st.caption("🟢 Low (<100)  🟡 Medium (100–300)  🔴 High (>300)")
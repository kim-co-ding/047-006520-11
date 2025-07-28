import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os

def load_csv(filename):
    try:
        return pd.read_csv(filename, encoding='utf-8-sig')
    except UnicodeDecodeError:
        import chardet
        with open(filename, 'rb') as f:
            enc = chardet.detect(f.read())['encoding']
        return pd.read_csv(filename, encoding=enc)

def train_models(df):
    df_res = df.dropna(subset=["Pitch(mm)", "Resistance(ohm)"])
    df_wt = df.dropna(subset=["Pitch(mm)", "Weight(g)"])

    X_res = df_res[["Pitch(mm)"]].values
    y_res = df_res["Resistance(ohm)"].values

    X_wt = df_wt[["Pitch(mm)"]].values
    y_wt = df_wt["Weight(g)"].values

    poly = PolynomialFeatures(degree=2)
    X_res_poly = poly.fit_transform(X_res)
    X_wt_poly = poly.fit_transform(X_wt)

    model_res = LinearRegression().fit(X_res_poly, y_res)
    model_wt = LinearRegression().fit(X_wt_poly, y_wt)

    return model_res, model_wt, poly, df_res, df_wt

def get_equation(model, poly):
    coeffs = model.coef_
    intercept = model.intercept_

    terms = [f"{intercept:.4f}"]
    if poly.degree >= 1 and len(coeffs) > 1:
        terms.append(f"{coeffs[1]:+.4f}·x")
    if poly.degree >= 2 and len(coeffs) > 2:
        terms.append(f"{coeffs[2]:+.4f}·x²")
    return " ".join(terms)

def out_of_range_warning(value, min_val, max_val, name):
    if value < min_val or value > max_val:
        return f"⚠️ {name} out of range.\n"
    return ""

def plot_graph(df_res, df_wt, model_res, model_wt, poly, min_pitch, max_pitch, pitch=None, res_pred=None, wt_pred=None):
    fig, ax1 = plt.subplots(figsize=(12, 6))  # Bigger plot

    ax2 = ax1.twinx()

    ax1.scatter(df_res["Pitch(mm)"], df_res["Resistance(ohm)"], color="blue", marker="x", label="Resistance Data")
    ax2.scatter(df_wt["Pitch(mm)"], df_wt["Weight(g)"], color="red", marker="x", label="Weight Data")

    x_vals = np.linspace(min_pitch, max_pitch, 200).reshape(-1, 1)
    x_poly = poly.transform(x_vals)
    ax1.plot(x_vals, model_res.predict(x_poly), color="blue", linestyle="--", label="Resistance Fit")
    ax2.plot(x_vals, model_wt.predict(x_poly), color="red", linestyle="--", label="Weight Fit")

    if pitch is not None and res_pred is not None and wt_pred is not None:
        ax1.scatter([pitch], [res_pred], color='blue', s=100, marker='o', label='Predicted Resistance')
        ax2.scatter([pitch], [wt_pred], color='red', s=100, marker='o', label='Predicted Weight')

    ax1.set_xlabel("Pitch (mm)")
    ax1.set_ylabel("Resistance (ohm)", color="blue")
    ax2.set_ylabel("Weight (g)", color="red")
    ax1.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='red')
    plt.title("Quadratic Regression of Resistance and Weight by Pitch")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower left', fontsize=8)

    st.pyplot(fig)

def main():
    st.set_page_config(layout="wide")  # Wide layout

    st.title("047-006520-11 Pitch, Resistance, Weight Regression")

    DATA_FILE = "data.csv"
    if not os.path.exists(DATA_FILE):
        st.error("data.csv must be in the same folder as this script.")
        return

    df = load_csv(DATA_FILE)
    df = df.rename(columns={"Resistance(Ω)": "Resistance(ohm)"})

    model_res, model_wt, poly, df_res, df_wt = train_models(df)

    min_pitch = df["Pitch(mm)"].dropna().min()
    max_pitch = df["Pitch(mm)"].dropna().max()

    res_eq = get_equation(model_res, poly)
    wt_eq = get_equation(model_wt, poly)

    st.markdown(f"**Pitch range:** {min_pitch:.4f} ~ {max_pitch:.4f}")

    col1, col2 = st.columns([1, 3])  # Input / Graph

    with col1:
        st.markdown("### Input")
        pitch_input = st.text_input("Pitch (mm)", "", key="pitch", help="Enter numeric pitch", max_chars=10)
        resistance_input = st.text_input("Resistance (ohm)", "", key="res", help="Enter numeric resistance", max_chars=10)
        weight_input = st.text_input("Weight (g)", "", key="wt", help="Enter numeric weight", max_chars=10)

        st.markdown("### Regression Equations")
        st.markdown(f"**Resistance (ohm)** = {res_eq}")
        st.markdown(f"**Weight (g)** = {wt_eq}")

        warn = ""
        pitch = resistance = weight = None

        def try_float(s):
            try:
                return float(s)
            except:
                return None

        pitch = try_float(pitch_input) if pitch_input else None
        resistance = try_float(resistance_input) if resistance_input else None
        weight = try_float(weight_input) if weight_input else None

        if (pitch_input and pitch is None) or (resistance_input and resistance is None) or (weight_input and weight is None):
            st.warning("⚠️ Please enter valid numeric values.")

        if st.button("Predict"):
            if not any([pitch, resistance, weight]):
                st.warning("⚠️ Please input at least one value.")
                return

            if pitch is not None and resistance is None and weight is None:
                x_input = poly.transform([[pitch]])
                resistance = model_res.predict(x_input)[0]
                weight = model_wt.predict(x_input)[0]
                warn += out_of_range_warning(pitch, min_pitch, max_pitch, "Pitch")
                st.success(f"{warn}Predicted:\nResistance = {resistance:.2f}\nWeight = {weight:.2f}")
                with col2:
                    plot_graph(df_res, df_wt, model_res, model_wt, poly, min_pitch, max_pitch, pitch, resistance, weight)

            elif resistance is not None and pitch is None and weight is None:
                pitches = np.linspace(min_pitch, max_pitch, 1000).reshape(-1, 1)
                res_preds = model_res.predict(poly.transform(pitches))
                idx = np.abs(res_preds - resistance).argmin()
                pitch = pitches[idx][0]
                weight = model_wt.predict(poly.transform([[pitch]]))[0]
                warn += out_of_range_warning(resistance, df_res["Resistance(ohm)"].min(), df_res["Resistance(ohm)"].max(), "Resistance")
                st.success(f"{warn}Estimated:\nPitch = {pitch:.4f}\nWeight = {weight:.2f}")
                with col2:
                    plot_graph(df_res, df_wt, model_res, model_wt, poly, min_pitch, max_pitch, pitch, resistance, weight)

            elif weight is not None and pitch is None and resistance is None:
                pitches = np.linspace(min_pitch, max_pitch, 1000).reshape(-1, 1)
                wt_preds = model_wt.predict(poly.transform(pitches))
                idx = np.abs(wt_preds - weight).argmin()
                pitch = pitches[idx][0]
                resistance = model_res.predict(poly.transform([[pitch]]))[0]
                warn += out_of_range_warning(weight, df_wt["Weight(g)"].min(), df_wt["Weight(g)"].max(), "Weight")
                st.success(f"{warn}Estimated:\nPitch = {pitch:.4f}\nResistance = {resistance:.2f}")
                with col2:
                    plot_graph(df_res, df_wt, model_res, model_wt, poly, min_pitch, max_pitch, pitch, resistance, weight)

            elif pitch is not None and resistance is not None and weight is None:
                pitch_candidates = np.linspace(max(min_pitch, pitch-0.005), min(max_pitch, pitch+0.005), 200).reshape(-1, 1)
                res_preds = model_res.predict(poly.transform(pitch_candidates))
                close_idx = np.abs(res_preds - resistance) < 0.1
                filtered = pitch_candidates[close_idx]
                if len(filtered) == 0:
                    idx_min = np.abs(res_preds - resistance).argmin()
                    filtered = pitch_candidates[idx_min:idx_min+1]
                wt_preds = model_wt.predict(poly.transform(filtered))
                weight_pred = np.mean(wt_preds)
                warn += out_of_range_warning(pitch, min_pitch, max_pitch, "Pitch")
                warn += out_of_range_warning(resistance, df_res["Resistance(ohm)"].min(), df_res["Resistance(ohm)"].max(), "Resistance")
                st.success(f"{warn}Predicted:\nWeight = {weight_pred:.2f}")
                with col2:
                    plot_graph(df_res, df_wt, model_res, model_wt, poly, min_pitch, max_pitch, pitch, resistance, weight_pred)

            elif pitch is not None and weight is not None and resistance is None:
                pitch_candidates = np.linspace(max(min_pitch, pitch-0.005), min(max_pitch, pitch+0.005), 200).reshape(-1, 1)
                wt_preds = model_wt.predict(poly.transform(pitch_candidates))
                close_idx = np.abs(wt_preds - weight) < 0.1
                filtered = pitch_candidates[close_idx]
                if len(filtered) == 0:
                    idx_min = np.abs(wt_preds - weight).argmin()
                    filtered = pitch_candidates[idx_min:idx_min+1]
                res_preds = model_res.predict(poly.transform(filtered))
                resistance_pred = np.mean(res_preds)
                warn += out_of_range_warning(pitch, min_pitch, max_pitch, "Pitch")
                warn += out_of_range_warning(weight, df_wt["Weight(g)"].min(), df_wt["Weight(g)"].max(), "Weight")
                st.success(f"{warn}Predicted:\nResistance = {resistance_pred:.2f}")
                with col2:
                    plot_graph(df_res, df_wt, model_res, model_wt, poly, min_pitch, max_pitch, pitch, resistance_pred, weight)

            elif resistance is not None and weight is not None and pitch is None:
                pitches = np.linspace(min_pitch, max_pitch, 1000).reshape(-1, 1)
                res_preds = model_res.predict(poly.transform(pitches))
                wt_preds = model_wt.predict(poly.transform(pitches))
                errors = (res_preds - resistance) ** 2 + (wt_preds - weight) ** 2
                best_idx = errors.argmin()
                pitch = pitches[best_idx][0]
                warn += out_of_range_warning(resistance, df_res["Resistance(ohm)"].min(), df_res["Resistance(ohm)"].max(), "Resistance")
                warn += out_of_range_warning(weight, df_wt["Weight(g)"].min(), df_wt["Weight(g)"].max(), "Weight")
                st.success(f"{warn}Estimated:\nPitch = {pitch:.4f}")
                with col2:
                    plot_graph(df_res, df_wt, model_res, model_wt, poly, min_pitch, max_pitch, pitch, resistance, weight)

            else:
                st.warning("⚠️ Please enter only one or two values.")

if __name__ == "__main__":
    main()

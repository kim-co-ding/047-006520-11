import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.widgets import TextBox, Button
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
    df_res = df.dropna(subset=["Pitch(mm)", "Resistance(Ω)"])
    df_wt = df.dropna(subset=["Pitch(mm)", "Weight(g)"])

    X_res = df_res[["Pitch(mm)"]].values
    y_res = df_res["Resistance(Ω)"].values

    X_wt = df_wt[["Pitch(mm)"]].values
    y_wt = df_wt["Weight(g)"].values

    poly = PolynomialFeatures(degree=2)
    X_res_poly = poly.fit_transform(X_res)
    X_wt_poly = poly.fit_transform(X_wt)

    model_res = LinearRegression().fit(X_res_poly, y_res)
    model_wt = LinearRegression().fit(X_wt_poly, y_wt)

    return model_res, model_wt, poly, df_res, df_wt

def out_of_range_warning(value, min_val, max_val, name):
    if value < min_val or value > max_val:
        return f"⚠️ {name} out of range.\n"
    return ""

DATA_FILE = "data.csv"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("data.csv must be in the same folder as this script.")

df = load_csv(DATA_FILE)
model_res, model_wt, poly, df_res, df_wt = train_models(df)

min_pitch = df["Pitch(mm)"].dropna().min()
max_pitch = df["Pitch(mm)"].dropna().max()

fig, ax1 = plt.subplots(figsize=(14, 6))
plt.subplots_adjust(left=0.08, right=0.65, top=0.9, bottom=0.2)
ax2 = ax1.twinx()

ax1.scatter(df_res["Pitch(mm)"], df_res["Resistance(Ω)"], color="blue", marker="x", label="Resistance Data")
ax2.scatter(df_wt["Pitch(mm)"], df_wt["Weight(g)"], color="red", marker="x", label="Weight Data")

x_vals = np.linspace(min_pitch, max_pitch, 200).reshape(-1, 1)
x_poly = poly.transform(x_vals)
ax1.plot(x_vals, model_res.predict(x_poly), color="blue", linestyle="--", label="Resistance Fit")
ax2.plot(x_vals, model_wt.predict(x_poly), color="red", linestyle="--", label="Weight Fit")

ax1.set_xlabel("Pitch (mm)")
ax1.set_ylabel("Resistance (Ω)", color="blue")
ax2.set_ylabel("Weight (g)", color="red")
plt.title("Quadratic  Regression of Resistance and Weight by Pitch")

res_eq = f"Resistance = {model_res.coef_[2]:.2f}·x² + {model_res.coef_[1]:.2f}·x + {model_res.intercept_:.2f}  (R²={model_res.score(poly.transform(df_res[['Pitch(mm)']]), df_res['Resistance(Ω)']):.3f})"
wt_eq = f"Weight = {model_wt.coef_[2]:.2f}·x² + {model_wt.coef_[1]:.2f}·x + {model_wt.intercept_:.2f}  (R²={model_wt.score(poly.transform(df_wt[['Pitch(mm)']]), df_wt['Weight(g)']):.3f})"

fig.text(0.7, 0.88, res_eq, color='blue', fontsize=9, ha='left')
fig.text(0.7, 0.82, wt_eq, color='red', fontsize=9, ha='left')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc='center left', bbox_to_anchor=(0.7, 0.7), fontsize=9)

textbox_pitch = TextBox(plt.axes([0.76, 0.56, 0.2, 0.05]), "Pitch:")
textbox_res = TextBox(plt.axes([0.76, 0.48, 0.2, 0.05]), "Resistance:")
textbox_wt = TextBox(plt.axes([0.76, 0.40, 0.2, 0.05]), "Weight:")

axbutton = plt.axes([0.76, 0.30, 0.15, 0.05])
button = Button(axbutton, "Predict")

result_ax = plt.axes([0.70, 0.20, 0.28, 0.09])
result_ax.axis("off")
result_text = result_ax.text(0, 0.5, "", fontsize=10, va="center")

res_point = ax1.scatter([], [], color='blue', s=100, marker='o')
wt_point = ax2.scatter([], [], color='red', s=100, marker='o')

def predict(event):
    pitch_str = textbox_pitch.text.strip()
    res_str = textbox_res.text.strip()
    wt_str = textbox_wt.text.strip()

    try:
        pitch = float(pitch_str) if pitch_str else None
        resistance = float(res_str) if res_str else None
        weight = float(wt_str) if wt_str else None
    except ValueError:
        result_text.set_text("⚠️ Invalid input.\nPlease enter valid numbers.")
        fig.canvas.draw_idle()
        return

    if not any([pitch_str, res_str, wt_str]):
        result_text.set_text("⚠️ Please input at least one value.")
        fig.canvas.draw_idle()
        return

    warn = ""

    if pitch is not None and resistance is None and weight is None:
        x_input = poly.transform([[pitch]])
        resistance = model_res.predict(x_input)[0]
        weight = model_wt.predict(x_input)[0]
        warn += out_of_range_warning(pitch, min_pitch, max_pitch, "Pitch")
        result_text.set_text(f"{warn}Predicted:\nResistance = {resistance:.2f}\nWeight = {weight:.2f}")
        res_point.set_offsets([[pitch, resistance]])
        wt_point.set_offsets([[pitch, weight]])

    elif resistance is not None and pitch is None and weight is None:
        pitches = np.linspace(min_pitch, max_pitch, 1000).reshape(-1, 1)
        res_preds = model_res.predict(poly.transform(pitches))
        idx = np.abs(res_preds - resistance).argmin()
        pitch = pitches[idx][0]
        weight = model_wt.predict(poly.transform([[pitch]]))[0]
        warn += out_of_range_warning(resistance, df_res["Resistance(Ω)"].min(), df_res["Resistance(Ω)"].max(), "Resistance")
        result_text.set_text(f"{warn}Estimated:\nPitch = {pitch:.4f}\nWeight = {weight:.2f}")
        res_point.set_offsets([[pitch, resistance]])
        wt_point.set_offsets([[pitch, weight]])

    elif weight is not None and pitch is None and resistance is None:
        pitches = np.linspace(min_pitch, max_pitch, 1000).reshape(-1, 1)
        wt_preds = model_wt.predict(poly.transform(pitches))
        idx = np.abs(wt_preds - weight).argmin()
        pitch = pitches[idx][0]
        resistance = model_res.predict(poly.transform([[pitch]]))[0]
        warn += out_of_range_warning(weight, df_wt["Weight(g)"].min(), df_wt["Weight(g)"].max(), "Weight")
        result_text.set_text(f"{warn}Estimated:\nPitch = {pitch:.4f}\nResistance = {resistance:.2f}")
        res_point.set_offsets([[pitch, resistance]])
        wt_point.set_offsets([[pitch, weight]])

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
        warn += out_of_range_warning(resistance, df_res["Resistance(Ω)"].min(), df_res["Resistance(Ω)"].max(), "Resistance")
        result_text.set_text(f"{warn}Predicted:\nWeight = {weight_pred:.2f}")
        res_point.set_offsets([[pitch, resistance]])
        wt_point.set_offsets([[pitch, weight_pred]])

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
        result_text.set_text(f"{warn}Predicted:\nResistance = {resistance_pred:.2f}")
        res_point.set_offsets([[pitch, resistance_pred]])
        wt_point.set_offsets([[pitch, weight]])

    elif resistance is not None and weight is not None and pitch is None:
        pitches = np.linspace(min_pitch, max_pitch, 1000).reshape(-1, 1)
        res_preds = model_res.predict(poly.transform(pitches))
        wt_preds = model_wt.predict(poly.transform(pitches))
        errors = (res_preds - resistance) ** 2 + (wt_preds - weight) ** 2
        best_idx = errors.argmin()
        pitch = pitches[best_idx][0]
        warn += out_of_range_warning(resistance, df_res["Resistance(Ω)"].min(), df_res["Resistance(Ω)"].max(), "Resistance")
        warn += out_of_range_warning(weight, df_wt["Weight(g)"].min(), df_wt["Weight(g)"].max(), "Weight")
        result_text.set_text(f"{warn}Estimated:\nPitch = {pitch:.4f}")
        res_point.set_offsets([[pitch, resistance]])
        wt_point.set_offsets([[pitch, weight]])

    else:
        result_text.set_text("⚠️ Please enter only one or two values.")

    fig.canvas.draw_idle()

button.on_clicked(predict)

plt.show()

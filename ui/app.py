import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'sleep_model.pkl')

model = joblib.load(MODEL_PATH)

label_map = {
    0: "Poor",
    1: "Average",
    2: "Good"
}

def predict_sleep():
    try:
        sleep = float(entry_sleep.get())
        activity = float(entry_activity.get())
        stress = float(entry_stress.get())
        steps = float(entry_steps.get())

        data = pd.DataFrame([{
            'Sleep Duration': sleep,
            'Physical Activity Level': activity,
            'Stress Level': stress,
            'Daily Steps': steps
        }])

        prediction = model.predict(data)[0]

        if prediction == 0:
            tips = "Improve physical activity, reduce stress, and maintain sleep routine."
        elif prediction == 1:
            tips = "Maintain consistency and slightly improve activity levels."
        else:
            tips = "Excellent habits! Keep maintaining your lifestyle."

        messagebox.showinfo(
            "Sleep Quality Prediction",
            f"Sleep Quality: {label_map[prediction]}\n\nTips: {tips}"
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ---------- UI ----------
root = tk.Tk()
root.title("Sleep Quality Predictor")
root.geometry("420x420")

tk.Label(root, text="Sleep Quality Predictor", font=("Arial", 16)).pack(pady=10)

tk.Label(root, text="Sleep Duration (hours)").pack()
entry_sleep = tk.Entry(root)
entry_sleep.pack()

tk.Label(root, text="Physical Activity Level (0–100)").pack()
entry_activity = tk.Entry(root)
entry_activity.pack()

tk.Label(root, text="Stress Level (0–10)").pack()
entry_stress = tk.Entry(root)
entry_stress.pack()

tk.Label(root, text="Daily Steps").pack()
entry_steps = tk.Entry(root)
entry_steps.pack()

tk.Button(
    root,
    text="Predict Sleep Quality",
    command=predict_sleep,
    bg="#4CAF50",
    fg="white"
).pack(pady=20)

root.mainloop()

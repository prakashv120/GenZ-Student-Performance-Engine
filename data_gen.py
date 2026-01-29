import pandas as pd
import numpy as np

np.random.seed(42)

data = []

for i in range(500):
    attendance = np.random.randint(40, 100)
    study_hours = np.random.randint(1, 8)
    internal_marks = np.random.randint(5, 30)
    assignments_done = np.random.randint(0, 10)
    sleep_hours = np.random.randint(4, 9)
    phone_usage = np.random.randint(1, 8)

    score = (attendance * 0.3 +
             study_hours * 5 +
             internal_marks * 1.5 +
             assignments_done * 2 -
             phone_usage * 2)

    if score > 120:
        result = "Excellent"
    elif score > 80:
        result = "Average"
    else:
        result = "Poor"

    data.append([
        attendance,
        study_hours,
        internal_marks,
        assignments_done,
        sleep_hours,
        phone_usage,
        result
    ])

columns = [
    "attendance",
    "study_hours",
    "internal_marks",
    "assignments_done",
    "sleep_hours",
    "phone_usage",
    "result"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("student_behavior.csv", index=False)

print("student_behavior.csv generated successfully!")

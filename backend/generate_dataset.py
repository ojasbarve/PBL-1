import pandas as pd
import numpy as np

np.random.seed(42)

num_sessions = 2000

data = {
    "session_id": range(1, num_sessions + 1),
    "total_clicks": np.random.randint(5, 50, num_sessions),
    "back_clicks": np.random.randint(0, 10, num_sessions),
    "retry_count": np.random.randint(0, 6, num_sessions),
    "dwell_time": np.random.randint(10, 600, num_sessions),  # seconds
    "page_switch_count": np.random.randint(1, 15, num_sessions)
}

df = pd.DataFrame(data)

# Friction labeling logic
df["friction_label"] = np.where(
    (df["retry_count"] > 2) |
    (df["back_clicks"] > 5) |
    (df["dwell_time"] > 400),
    1,
    0
)

df.to_csv("sessions.csv", index=False)

print("Dataset generated successfully!")
print(df.head())
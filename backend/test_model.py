import joblib
import numpy as np

model = joblib.load("friction_model.pkl")

# Example new session
new_session = np.array([[30, 7, 3, 500, 10]])

prediction = model.predict(new_session)
probability = model.predict_proba(new_session)

print("Prediction (1 = High Friction, 0 = Low):", prediction[0])
print("Friction Probability:", probability[0][1])
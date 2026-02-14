import joblib

# Load the trained pipeline
pipeline = joblib.load("outputs/pipeline.joblib")

# Example news to classify
samples = [
    "The president announces new healthcare policies.",
    "Scientists discovered a unicorn in Africa!"
]

# Predict probabilities
probs = pipeline.predict_proba(samples)
preds = (probs[:, 1] >= 0.5).astype(int)

# Print results
for text, pred, prob in zip(samples, preds, probs[:, 1]):
    label = "FAKE" if pred == 1 else "REAL"
    print(f"\nNews: {text}\nPrediction: {label}, Probability of FAKE: {prob:.2f}")


import joblib

# Load the trained pipeline
pipeline = joblib.load("outputs/pipeline.joblib")

# List of news items to test
news_list = [
    "Aliens have landed in New York City and are meeting with world leaders.",
    "The local government announces a new public park in the city center next month.",
    "Scientists invent a pill that makes humans invisible for 24 hours!",
    "The city council approves a new library renovation project starting next week.",
    "Time travel experiments succeed: People from 3026 visit present-day New York!",
    "Local farmers report record wheat harvest this season.",
    "Dogs in the city learn to talk using AI collars."
]

# Make predictions
for news in news_list:
    prob_fake = pipeline.predict_proba([news])[0][1]  # Probability of FAKE
    prediction = "FAKE" if prob_fake >= 0.5 else "REAL"
    print(f"News: {news}")
    print(f"Prediction: {prediction}, Probability of FAKE: {prob_fake:.2f}\n")

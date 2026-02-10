import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# --- Streamlit UI ---
st.title("Spam Email Detector")

# Sample dataset
emails = [
    "Win a free iPhone now", "Meeting at 11 am tomorrow",
    "Congratulations you won lottery", "Project discussion with team",
    "Claim your prize immediately", "Please find the attached report",
    "Limited offer buy now", "Urgent offer expires today",
    "Schedule the meeting for Monday", "You have won a cash prize",
    "Monthly performance report attached", "Exclusive deal just for you"
]
labels = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # 1 = Spam, 0 = Not Spam

# Vectorization
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=1
)
X = vectorizer.fit_transform(emails)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

# Train model
model = LinearSVC(C=1.0, random_state=42)
model.fit(X_train, y_train)

# Accuracy
st.write(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")

# User input
user_msg = st.text_area("Enter Email Message")
if st.button("Check"):
    msg_vec = vectorizer.transform([user_msg])
    pred = model.predict(msg_vec)[0]
    st.write("Result: **Spam Email**" if pred == 1 else "Result: **Not Spam**")

# --- Visualization: Top Features ---
if st.checkbox("Show Important Features"):
    # Get feature weights
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_.flatten()   # FIXED: no .toarray()

    # Top positive (spam) and negative (not spam) features
    top_n = 10
    top_spam_idx = np.argsort(coefs)[-top_n:]
    top_notspam_idx = np.argsort(coefs)[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_names[top_spam_idx], coefs[top_spam_idx], color="red", label="Spam indicators")
    ax.barh(feature_names[top_notspam_idx], coefs[top_notspam_idx], color="green", label="Not Spam indicators")
    ax.set_title("Top Features Influencing Spam Detection")
    ax.legend()
    st.pyplot(fig)

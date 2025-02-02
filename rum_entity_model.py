import joblib
import sys

# Load trained models
context_model = joblib.load("models/context_model.pkl")
server_model = joblib.load("models/server_model.pkl")
software_model = joblib.load("models/software_model.pkl")

def extract_entities(summary, description):
    text = summary + " " + description

    return {
        "context": context_model.predict([text])[0],
        "server": server_model.predict([text])[0],
        "software": software_model.predict([text])[0]
    }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_model.py '<summary>' '<description>'")
        sys.exit(1)

    summary = sys.argv[1]
    description = sys.argv[2]

    result = extract_entities(summary, description)
    print("Extracted Entities:", result)

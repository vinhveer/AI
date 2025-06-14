import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), 'ecommerce_multiturn.json')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'chatbot_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

# Create model directory if not exists
def ensure_model_dir():
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)

# Load dataset from JSON file
def load_data(path=DATA_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        sessions = json.load(f)
    questions, answers = [], []
    for session in sessions:
        for msg in session.get('conversations', []):
            if msg.get('from') == 'human':
                # next message from GPT as the answer
                questions.append(msg.get('value', '').strip())
                # find following gpt reply
                idx = session['conversations'].index(msg) + 1
                if idx < len(session['conversations']):
                    ans = session['conversations'][idx]
                    answers.append(ans.get('value', '').strip())
                else:
                    answers.append('')
    return questions, answers

# Train model and save artifacts
def train_and_save():
    print('Loading data...')
    questions, answers = load_data()
    print(f'Loaded {len(questions)} QA pairs.')

    print('Vectorizing questions...')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)

    print('Training KNN classifier...')
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, answers)

    print('Saving model and vectorizer...')
    ensure_model_dir()
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print('Training complete.')

# Reply function
def chatbot_reply(user_input):
    # Load artifacts
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    vec = vectorizer.transform([user_input])
    return model.predict(vec)[0]

if __name__ == '__main__':
    train_and_save()
    # Optional: interactive chat
    print("Chatbot ready. Type 'exit' to quit.")
    while True:
        msg = input('You: ').strip()
        if msg.lower() in ('exit', 'quit'):
            print('Bot: Bye!')
            break
        reply = chatbot_reply(msg)
        print('Bot:', reply)

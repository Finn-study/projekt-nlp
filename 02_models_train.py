import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Download von NLTK-Ressourcen (einmalig notwendig)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# 2. Daten laden
file_path = "pfad\\final_dataset.csv"
df = pd.read_csv(file_path)
print("Daten erfolgreich geladen. Infos zum DataFrame:")
df.info()

# Sicherheitsmaßnahme: Entferne Zeilen, bei denen Text oder Titel fehlen könnten
df.dropna(subset=['title', 'text'], inplace=True)

# Definiere Features (X) und Zielvariable (y)
X = df['title'] + ' ' + df['text']
y = df['rating']


# 3. Aufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Größe der Trainingsdaten (X):", X_train.shape)
print("Größe der Testdaten (X):", X_test.shape)


# 4. Textvorverarbeitung und TF-IDF-Vektorisierung
def lemmatizing_tokenizer(text):
    """
    Diese Funktion erhält eine Rezension als Eingabe und führt die komplette Verarbeitung durch:
    Tokenisierung, Entfernen von Stoppwörtern und Nicht-Buchstaben & Lemmatisierung
    """
    lemmatizer = WordNetLemmatizer()
    # 1. Nur Buchstaben behalten und in Kleinbuchstaben umwandeln
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())

    # 2. Tokenisierung
    tokens = nltk.word_tokenize(text)

    # 3. Lemmatisierung und Stoppwort-Entferunung
    stop_words = set(stopwords.words('english')) # Englisch, da der Datensatz nur englische Rezensionen beinhaltet
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]

    return lemmatized_tokens

# Erstelle den Vektorisierer
vectorizer = TfidfVectorizer(
    tokenizer=lemmatizing_tokenizer,
    max_features=5000 # Begrenzung auf die 5.000 häufigsten Wörter
)

# Vokabular von den Trainingsdaten lernen und diese transformieren
X_train_vec = vectorizer.fit_transform(X_train)

# Testdaten mit gelernten Vokabular transformieren
X_test_vec = vectorizer.transform(X_test)

print("Vektorisierung abgeschlossen.")
print("Form der Trainingsdaten-Matrix:", X_train_vec.shape)
print("Form der Testdaten-Matrix:", X_test_vec.shape)
print("\nVerteilung der Bewertungen im Trainings-Datensatz:")
print(y_train.value_counts().sort_index())
print("\nVerteilung der Bewertungen im Test-Datensatz:")
print(y_test.value_counts().sort_index())



#####################################################
# MODELL 1: MULTINOMIAL NAIVE BAYES (5 Klassen)
#####################################################
print("Training: Multinomial Naive Bayes (5 Klassen)")
model_nb_5 = MultinomialNB()
model_nb_5.fit(X_train_vec, y_train)
print("Modelltraining abgeschlossen.")

# Evaluation
y_pred_nb_5 = model_nb_5.predict(X_test_vec)
accuracy_nb_5 = accuracy_score(y_test, y_pred_nb_5)
print(f"Accuracy Naive Bayes (5 Klassen): {accuracy_nb_5:.4f}")

# Konfusionsmatrix
cm_nb_5 = confusion_matrix(y_test, y_pred_nb_5, labels=[1.0, 2.0, 3.0, 4.0, 5.0])
plt.figure(figsize=(10, 7))
sns.heatmap(cm_nb_5, annot=True, fmt='d', cmap='Blues',
            xticklabels=[1, 2, 3, 4, 5],
            yticklabels=[1, 2, 3, 4, 5])
plt.xlabel('Vorhergesagte Bewertung')
plt.ylabel('Tatsächliche Bewertung')
plt.show()

# Evaluation nach Produktkategorien
# Temporäre DataFrames, um die Kategorien zuzuordnen
train_df = pd.DataFrame({'category': df.loc[y_train.index, 'category']})
test_df = pd.DataFrame({'category': df.loc[y_test.index, 'category']})

# Aufstellung der Anzahl der Rezensionen je Kategorie
print("Verteilung der Kategorien im Datensatz:")
print(df['category'].value_counts())
print("Verteilung der Kategorien im Trainings-Datensatz:")
print(train_df['category'].value_counts())
print("Verteilung der Kategorien im Test-Datensatz:")
print(test_df['category'].value_counts())

# Neuer DataFrame mit den Testergebnissen und Kategorien
results_df = pd.DataFrame({ 
    'true_rating': y_test,
    'predicted_rating': y_pred_nb_5,
    'category': df.loc[y_test.index, 'category']
})

# Gruppierung nach Kategorie
accuracy_per_category = results_df.groupby('category').apply(
    lambda x: accuracy_score(x['true_rating'], x['predicted_rating'])
)

print("Genauigkeit pro Kategorie:")
print(accuracy_per_category)



#####################################################
# MODELL 2: MULTINOMIAL NAIVE BAYES (3 Klassen)
#####################################################
print("Training: Multinomial Naive Bayes (3 Klassen)")

def change_rating(rating):
    if rating <= 2:
        return 0 # Negativ
    elif rating ==3:
        return 1 # Neutral
    else:
        return 2 # Positiv

y_train_sentiment = y_train.map(change_rating)
y_test_sentiment = y_test.map(change_rating)

model_nb_3 = MultinomialNB()
model_nb_3.fit(X_train_vec, y_train_sentiment)
print("Modelltraining abgeschlossen")

# Evaluation
y_pred_nb_3 = model_nb_3.predict(X_test_vec)
accuracy_nb_3 = accuracy_score(y_test_sentiment, y_pred_nb_3)
print(f"Accuracy Naive Bayes (3 Klassen): {accuracy_nb_3:.4f}")

# Konfusionsmatrix
cm_nb_3 = confusion_matrix(y_test_sentiment, y_pred_nb_3)
sentiment_labels = ['Negativ', 'Neutral', 'Positiv']
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb_3, annot=True, fmt='d', cmap='Blues',
            xticklabels=sentiment_labels,
            yticklabels=sentiment_labels)
plt.xlabel('Vorhergesagtes Sentiment')
plt.ylabel('Tatsächliches Sentiment')
plt.show()



#####################################################
# VORBEREITUNG FÜR NEURONALE NETZE
#####################################################
print("Vorbereitung der Daten für neuronale Netze")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_vec.toarray())
X_test_scaled = scaler.transform(X_test_vec.toarray())
print ("Daten erfolgreich skaliert.")



#####################################################
# MODELL 3: NEURONALES NETZ (5 Klassen)
#####################################################
print("Training: Neuronales Netz (5 Klassen)")
# Umwandlung der y-Werte in einem Vektor mit 5 Stellen
y_train_nn_5 = to_categorical(y_train -1, num_classes=5)
y_test_nn_5 = to_categorical(y_test - 1, num_classes=5)

# Neuronales Netz
model_nn_5 = Sequential([
    # Hidden Layer mit 128 Neuronen, sowie Definition des Input Layers
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5), # Zur Vermeidung von Overfitting: 50 % der Neuronen werden in dieser Schicht ignoriert

    # Hidden Layer mit 64 Neuronen
    Dense(64, activation='relu'),
    Dropout(0.5), # Zur Vermeidung von Overfitting: 50 % der Neuronen werden in dieser Schicht ignoriert

    # Output Layer mit 5 Neuronen, für jede Klasse einen
    Dense(5, activation='softmax') # Softmax berechnet die Wahrscheinlichkeit jeder Klasse
])

# Optimizer, Loss-Funktion und Accuracy-Metrik
model_nn_5.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model_nn_5.summary()

# Training des Modells
history_nn_5 = model_nn_5.fit(
    X_train_scaled, y_train_nn_5,
    epochs = 5, # Anzahl Trainingsdurchgänge
    batch_size=128, # Anzahl der auf einmal zu verarbeitenden Rezensionen
    validation_data=(X_test_scaled, y_test_nn_5),
    verbose=1 # Ausgabe der Ergebnisse nach jeder Epoche
)

# Evaluation
loss_nn_5, accuracy_nn_5 = model_nn_5.evaluate(X_test_scaled, y_test_nn_5, verbose=0)
print(f"\nAccuracy Neuronales Netz (5 Klassen): {accuracy_nn_5:.4f}")

# Konfusionsmatrix
y_pred_proba_nn_5 = model_nn_5.predict(X_test_scaled)
y_pred_nn_5 = np.argmax(y_pred_proba_nn_5, axis=1) + 1 # Addition um 1 um wieder die Skala 1 - 5 zu erhalten

cm_nn_5 = confusion_matrix(y_test, y_pred_nn_5, labels=[1, 2, 3, 4, 5])
plt.figure(figsize=(10, 7))
sns.heatmap(cm_nn_5, annot=True, fmt='d', cmap='Blues',
xticklabels=[1, 2, 3, 4, 5],
yticklabels=[1, 2, 3, 4, 5])
plt.xlabel('Vorhergesagte Bewertung')
plt.ylabel('Tatsächliche Bewertung')
plt.show()



#####################################################
# MODELL 4: NEURONALES NETZ (3 Klassen)
#####################################################
# Anwendung der Funktion auf y-Werte
y_train_nn_3 = to_categorical(y_train_sentiment, num_classes=3)
y_test_nn_3 = to_categorical(y_test_sentiment, num_classes=3)

# Neuronales Netz
model_nn_3 = Sequential([
    # Input Layer mit 128 Neuronen
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5), # Zur Vermeidung von Overfitting: 50 % der Neuronen werden in dieser Schicht ignoriert

    # Hidden Layer mit 64 Neuronen
    Dense(64, activation='relu'),
    Dropout(0.5), # Zur Vermeidung von Overfitting: 50 % der Neuronen werden in dieser Schicht ignoriert

    # Output Layer mit 3 Neuronen, für jede Klasse einen
    Dense(3, activation='softmax') # Softmax berechnet die Wahrscheinlichkeit jeder Klasse
])

# Optimizer, Loss-Funktion und Accuracy-Metrik
model_nn_3.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model_nn_3.summary()

# Training des Modells
history_nn_3 = model_nn_3.fit(
    X_train_scaled, y_train_nn_3,
    epochs = 5, # Anzahl Trainingsdurchgänge
    batch_size=128, # Verarbeitet immer 128 Rezensionen auf einmal
    validation_data=(X_test_scaled, y_test_nn_3),
    verbose=1 # Ausgabe der Ergebnisse nach jeder Epoche
)

# Evaluation
loss_nn_3, accuracy_nn_3 = model_nn_3.evaluate(X_test_scaled, y_test_nn_3, verbose=0)
print(f"\nAccuracy Neuraonales Netz (3 Klassen): {accuracy_nn_3:.4f}")

# Konfusionsmatrix
y_pred_proba_nn_3 = model_nn_3.predict(X_test_scaled)
y_pred_nn_3 = np.argmax(y_pred_proba_nn_3, axis=1)

cm_nn_3 = confusion_matrix(y_test_sentiment, y_pred_nn_3)
sentiment_labels = ['Negativ', 'Neutral', 'Positiv']

plt.figure(figsize=(8, 6))
sns.heatmap(cm_nn_3, annot=True, fmt='d', cmap='Blues',
            xticklabels=sentiment_labels,
            yticklabels=sentiment_labels)
plt.xlabel('Vorhergesagtes Sentiment')
plt.ylabel('Tatsächliches Sentiment')
plt.show()
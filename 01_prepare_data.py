import pandas as pd
import random
import gzip
import json

def sample_jsonl(input_path, num_samples):
    """
    Liest eine zufällige Stichprobe aus einer großen .jsonl.gz Datei
    und gibt sie als pandas DataFrame zurück
    """
    print(f"Starte Verarbeitung für: {input_path}")

    # Zählt die Zeilen
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        total_lines =sum(1 for line in f)

    # Random Seed sorgt für eine exakt gleiche zufällige Stichprobe
    random.seed(42)
    sample_indices = set(random.sample(range(total_lines), num_samples))

    # Datei lesen und nur ausgewählte Zeilen speichern
    sampled_data = []
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i in sample_indices:
                sampled_data.append(json.loads(line))

    # Rückgabe des Dataframe
    return pd.DataFrame(sampled_data)


# Pfade für alle Kategorien
input_paths = {
    "Automotive"    : "pfad\\Automotive.jsonl.gz",
    "Books"         : "pfad\\Books.jsonl.gz",
    "Video_Games"   : "pfad\\Video_Games.jsonl.gz"
}
final_output_path  = "pfad\\final_dataset.csv"
n_samples   = 150000 #Größe der Stichprobe je Kategorie


# Funktion für alle 3 Kategorien ausführen
all_dfs = []
for category_name, path in input_paths.items():
    df_sample = sample_jsonl(input_path=path, num_samples=n_samples)

    df_sample['category'] = category_name # Fügt die entsprechende Kategorie zu jeder Zeile als Spalte hinzu

    all_dfs.append(df_sample)


# Kombiniere die einzelnen DataFrames
df_final = pd.concat(all_dfs, ignore_index=True)


# 4. Filtern nach Mindesttextlänge
print(f"Originalgröße: {len(df_final)} Zeilen")
min_word_count = 5
df_final = df_final[df_final['text'].str.split().str.len() >= min_word_count].copy()
print(f"Größe nach Filterung (mind. {min_word_count} Wörter): {len(df_final)} Zeilen")


# 5. Finalen DataFrame speichern
print(f"Speichere finalen DataFrame mit {len(df_final)} Zeilen nach '{final_output_path}'...")
df_final.to_csv(final_output_path, index=False)
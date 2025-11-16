import pandas as pd

def preprocess_input_features(song):

    df = pd.DataFrame([song.dict()])

    # GENERAR FEATURES DERIVADAS (8 columnas nuevas)

    # 1. duration_min → duración en minutos
    df["duration_min"] = df["duration_ms"] / 60000

    # 2. beat_density → densidad de beats reales
    df["beat_density"] = df["tempo"] / df["duration_min"]

    # 3. energy_valence → energía × valence
    df["energy_valence"] = df["energy"] * df["valence"]

    # 4. dance_energy → danceability × energy
    df["dance_energy"] = df["danceability"] * df["energy"]

    # 5. speech_valence → speechiness × valence
    df["speech_valence"] = df["speechiness"] * df["valence"]

    # 6. acoustic_energy → acousticness × energy
    df["acoustic_energy"] = df["acousticness"] * df["energy"]

    # 7. inst_energy → instrumentalness × energy
    df["inst_energy"] = df["instrumentalness"] * df["energy"]

    # 8. dance_valence → danceability × valence
    df["dance_valence"] = df["danceability"] * df["valence"]

    ordered_columns = [
        "genre",
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",

        # Features derivadas:
        "duration_min",
        "beat_density",
        "energy_valence",
        "dance_energy",
        "speech_valence",
        "acoustic_energy",
        "inst_energy",
        "dance_valence"
    ]

    df = df[ordered_columns]

    return df

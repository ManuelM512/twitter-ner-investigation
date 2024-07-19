from transformers import pipeline
import pandas as pd

# Acá configuro mi modelo a utilizar
nlp_ner = pipeline(
    "ner",
    model="mrm8488/bert-spanish-cased-finetuned-ner",
    tokenizer=("mrm8488/bert-spanish-cased-finetuned-ner", {"use_fast": False}),
)


# A REVISAR: Para procesar los tweets, se guardan en una lista de id's, entidades y localización
def predicted_with_id(text, id, geo):
    ner_list = []
    for n_entity in nlp_ner(text):
        if (n_entity["word"] != "[UNK]") and (
            n_entity["entity"] == "I-LOC" or n_entity["entity"] == "B-LOC"
        ):
            if n_entity["word"][0] == "#":
                if len(ner_list) > 0:
                    ner_list[-1] = ner_list[-1] + n_entity["word"][2:]
            else:
                ner_list.append(n_entity["word"])
    if len(ner_list) > 0:
        location = ", ".join(ner_list)
        if geo != {}:
            id_entities_tweet = [id, location, geo["place_id"]]
        else:
            id_entities_tweet = [id, location, None]
        return id_entities_tweet


# Carga los dataframes, configura columnas e índices, y los devuelve en una lista
def load_dataframes():
    # Cargo los DataFrames
    twitter_doc = "../../../twitter_integration_twitter_collection(1).json"
    places_doc = "../../../twitter_integration_places_collection.json"
    twitter_df = pd.read_json(twitter_doc, lines=True)
    places_df = pd.read_json(places_doc)

    # Tomo solo unos pocos tweets, para probar en un dataset chico y no en 10k
    # twitter_df_limited = twitter_df.loc[:10, :]
    # Me quedo con las columnas que me sirven
    twitter_df_limited = twitter_df.loc[:, ["text", "id", "geo"]]
    places_df = places_df.loc[:, ["id", "full_name"]]
    # Renombro esta columna para que se llamen igual en ambos DF
    places_df = places_df.rename(columns={"id": "place_id"})
    # Cambio el índice para luego utilizarlo en el join
    places_df = places_df.set_index("place_id")

    # Esto es para ver todas las columnas en consola y que no oculte ninguna
    pd.set_option("display.max_columns", None)
    # Deslimita la cantidad de caracteres que se muestran de una columna
    pd.options.display.max_colwidth = 550

    return [places_df, twitter_df_limited]


# Devuelve una lista, con el dataframe de las entidades extraídas con geo, y las sin
def extract_places(places_df, twitter_df):
    twitter_df_limited = twitter_df

    # Proceso los datos, solo tomando las columnas de texto, id y localización
    result = [
        predicted_with_id(x, y, z)
        for x, y, z in zip(
            twitter_df_limited["text"],
            twitter_df_limited["id"],
            twitter_df_limited["geo"],
        )
    ]
    # Descarto aquellos que no contaban con ninguna entidad de localización reconocible
    result = list(filter(lambda item: item is not None, result))
    # Guardo los datos en un diccionario para luego convertirlo en un DataFrame
    ner_tweets = {
        "tweet_id": [tweet_id[0] for tweet_id in result],
        "place_name": [place_name[1] for place_name in result],
        "place_id": [place_id[2] for place_id in result],
    }
    ner_tweets_df = pd.DataFrame(data=ner_tweets)
    # Los que no tienen localización, los llevo a otro DF
    ner_tweets_no_geo_df = ner_tweets_df.loc[ner_tweets_df["place_id"].isnull()]
    ner_tweets_df = ner_tweets_df.loc[ner_tweets_df["place_id"].notnull()]
    # Join de los dos dataframes, mediante los id de lugar
    ner_with_places = ner_tweets_df.join(places_df, how="inner", on="place_id")
    return [ner_with_places, ner_tweets_no_geo_df]


# Compara las entidades extraídas con la localización de twitter
def compare_ner_2_geo(tweet_id, place_name, full_name):
    target = 0
    # Tremenda terrajeada, pero a veces lo pone en inglés o español, con esto se resuelve creo
    if full_name.find("España") != -1:
        full_name += " Spain"
    elif full_name.find("Spain") != -1:
        full_name += " España"
    entities_recognized_list = place_name.split(", ")
    if len(entities_recognized_list) > 1:
        for entity in entities_recognized_list:
            if full_name.find(entity) != -1:
                target = 1
    else:
        if full_name.find(place_name) != -1:
            target = 1
    return [tweet_id, target]


# Devuelve el DataFrame con los datos del tweet y el resultado de la comparación de ner / geo
def comparison_result(twitter_df, ner_df_list):
    tweets_dataframe = twitter_df
    ner_list = ner_df_list
    ner_with_places = ner_list[0]

    result = [
        compare_ner_2_geo(x, y, z)
        for x, y, z in zip(
            ner_with_places["tweet_id"],
            ner_with_places["place_name"],
            ner_with_places["full_name"],
        )
    ]
    tweet_id_target_data = {
        "tweet_id": [tweet_id[0] for tweet_id in result],
        "target": [target[1] for target in result],
    }
    dataframe_target = pd.DataFrame(tweet_id_target_data)
    dataframe_target = dataframe_target.set_index("tweet_id")
    ner_with_places = ner_with_places.set_index("tweet_id")
    ner_and_target = ner_with_places.join(dataframe_target, on="tweet_id", how="inner")

    # Variables numéricas para imprimir
    ner_with_places_count = len(ner_with_places.index)
    ner_without_places_count = len(ner_list[1].index)
    failed_tweets = len(ner_and_target.loc[ner_and_target.target == 0])
    correct_tweets = len(ner_and_target.loc[ner_and_target.target == 1])

    print(
        "\nCantidad de tweets analizados: ",
        len(tweets_dataframe.index),
        "\nCantidad de tweets con entidades reconocibles de lugar: ",
        ner_with_places_count + ner_without_places_count,
        "\nCantidad de tweets con localización activada: ",
        ner_with_places_count,
        "\nCantidad de tweets con localización desactivada: ",
        ner_without_places_count,
        "\nCantidad de tweets acertados (localización activada): ",
        correct_tweets,
        "\nCantidad de tweets no acertados (localización activada): ",
        failed_tweets,
        "\nPorcentaje de acierto (localización activada): {0:.2f}".format(
            (correct_tweets * 100) / ner_with_places_count
        ),
        "%",
    )
    return ner_and_target


# Toma los tweets sin geo, ya pasados por el ner, y los guarda en un archivo csv
def no_geo_to_csv(ner_no_geo_dataframe, twitter_df):
    ner_no_geo = ner_no_geo_dataframe
    df_texts = twitter_df
    df_texts = df_texts.loc[df_texts.geo == {}]
    df_texts = df_texts.loc[:, ["text", "id"]]
    df_texts = df_texts.rename(columns={"id": "tweet_id"})
    df_texts = df_texts.set_index("tweet_id")
    ner_no_geo = ner_no_geo.set_index("tweet_id")
    ner_no_geo = ner_no_geo.join(df_texts, how="inner", on="tweet_id")
    ner_no_geo.loc[:, ["text", "place_name"]].to_csv("tweets_no_geo.csv")


# La tengo para hacer pruebas rápidas de buscar un tweet o lugar
def prueba():
    dataframe_list = load_dataframes()
    places_df = dataframe_list[0]
    twitter_df_limited = dataframe_list[1]

    print(twitter_df_limited.loc[twitter_df_limited.id == 1623755053025619968].text)
    # print(places_df.loc[places_df.place_id == "63b8ba114723a129"])


# Cargar dataframes y pasar datos desde aca, poner como parámetro también en las funciones
def main():
    raw_dataframes_list = load_dataframes()
    places_df = raw_dataframes_list[0]
    twitter_df = raw_dataframes_list[1]

    ner_tweets_dataframe_list = extract_places(places_df, twitter_df)
    ner_without_geo = ner_tweets_dataframe_list[1]

    comparison_result(twitter_df, ner_tweets_dataframe_list)
    no_geo_to_csv(ner_without_geo, twitter_df)


main()

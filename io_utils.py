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

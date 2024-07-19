from transformers import pipeline

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

def comparison_result_to_string(result_dict):
    return (
        "\nCantidad de tweets analizados: ",
        result_dict["total_tweets"],
        "\nCantidad de tweets con entidades reconocibles de lugar: ",
        result_dict["ner_with_places_count"] + result_dict["ner_without_places_count"],
        "\nCantidad de tweets con localización activada: ",
        result_dict["ner_with_places_count"],
        "\nCantidad de tweets con localización desactivada: ",
        result_dict["ner_without_places_count"],
        "\nCantidad de tweets acertados (localización activada): ",
        result_dict["correct_tweets"],
        "\nCantidad de tweets no acertados (localización activada): ",
        result_dict["failed_tweets"],
        "\nPorcentaje de acierto (localización activada): {0:.2f}".format(
            result_dict["accuracy_percentage"]
        ),
        "%",
    )

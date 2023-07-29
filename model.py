from transformers import (
    TextClassificationPipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from preprocess import cleanse
from postprocess import postprocess


def predict_from_text(input_text):
    ## Load model and create pipeline
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "./saved_model/trained_models/bert-base-uncased-tutorial"
    )
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None)

    ## Clean title and get predicted tags
    clean_title = cleanse(input_text)
    model_output = pipe(clean_title)

    prediction = postprocess(model_output)

    if len(prediction) == 0:
        predict_output = "No matching tags."
    else:
        predict_output = ", ".join(prediction)

    return predict_output

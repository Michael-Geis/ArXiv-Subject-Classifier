import json


def postprocess(model_output, threshold_probability):
    with open("./data/arxiv-label-dict.json", "r") as file:
        subject_dict = json.loads(file.read())

    predicted_tags = [
        result["label"]
        for result in model_output[0]
        if result["score"] > threshold_probability
    ]

    return sorted([subject_dict[tag] for tag in predicted_tags])


# class ModelOutputDecoder(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         if y is None:
#             return X

#         ## Load label dictionary
#         with open("./data/arxiv-label-dict.json") as file:
#             string_dict = file.read()
#             label_dict = json.loads(string_dict)
#             col_list = list(label_dict.keys())

#         def decode_label(label):
#             ## For a row of y (individual label) returns the list of english subjects corresponding to this label
#             return [label_dict[col_list[index]] for index in np.where(label == 1)[0]]

#         num_rows, _ = y.shape

#         decoded_labels = []
#         for i in range(num_rows):
#             decoded_labels.append(decode_label(y[i, :]))

#         decoded_labels_as_series = pd.Series(
#             decoded_labels, name="decoded_labels", index=X.index
#         )

#         return pd.merge(
#             left=X,
#             left_index=True,
#             right=decoded_labels_as_series,
#             right_index=True,
#             validate="1:1",
#         )

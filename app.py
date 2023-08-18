import gradio as gr
import arxiv
import json
from model import predict_from_text

with open("./data/arxiv-label-dict.json", "r") as file:
    subject_dict = json.loads(file.read())


def parse_id(input_id):
    ## Grab article title and true categories from arXiv
    search = arxiv.Search(id_list=[input_id], max_results=1)
    result = next(search.results())
    raw_categories = result.categories
    title = result.title
    abstract = result.summary
    subject_tags = ", ".join(
        sorted(
            [subject_dict[tag] for tag in raw_categories if tag in subject_dict.keys()]
        )
    )

    return (title, subject_tags, abstract)


# def parse_title(input_title):
#     query_title = input_title.replace(" ", "\%20")
#     search = arxiv.Search(
#         query=f"ti:\%22{query_title}\%22",
#         sort_by=arxiv.SortCriterion.Relevance,
#         sort_order=arxiv.SortOrder.Descending,
#         max_results=1,
#     )
#     result = next(search.results())
#     raw_categories = result.categories
#     title = result.title

#     with open("./data/arxiv-label-dict.json", "r") as file:
#         subject_dict = json.loads(file.read())

#     subject_tags = ", ".join(
#         sorted(
#             [subject_dict[tag] for tag in raw_categories if tag in subject_dict.keys()]
#         )
#     )

#     return (title, subject_tags)


def outputs_from_id(input_id, threshold_probability):
    title, true_tags, abstract = parse_id(input_id)
    predicted_tags = predict_from_text(title, threshold_probability)
    return title, predicted_tags, true_tags, abstract


# def outputs_from_title(input_title, threshold_probability):
#     title, true_tags = parse_title(input_title)
#     predicted_tags = predict_from_text(title, threshold_probability)

#     return title, predicted_tags, true_tags


with gr.Blocks() as demo:
    gr.Markdown(
        """# <center> arXiv Subject Classifier
                This demo labels an arXiv math publication with appropriate subject categories based on its title. To get subject tag predictions for an article, input its unique
                arXiv ID (the 9 digit number appearing at the end of an article's URL). You can find the ID number of an article by searching for it on <a href="arxiv.org">arxiv.org</a>.
                For each possible tag, the model calculates the probability that the tag is applicable. Use the slider bar to set the minimum probability required for a tag to be predicted.
                For instance, when the slider is set to 0.50, only tags with a predicted probability of relevance over 50% will be suggested.
                </center>"""
    )
    with gr.Row():
        id_input = gr.Textbox(label="arXiv ID:", placeholder="XXXX.XXXXX")
        id_title = gr.Textbox(label="Title of Input Article:")
        id_predict = gr.Textbox(label="Predicted Subject Tags:")
        id_true = gr.Textbox(label="Actual Subject Tags:")
    threshold_probability = gr.Slider(
        label="Minimum Confidence For Tag Prediction:", value=0.5, minimum=0, maximum=1
    )
    id_button = gr.Button("Get Predicted Subject Tags")
    gr.Examples(
        label="Try These Example Articles:",
        examples=[
            "1709.07343",
            "2107.05105",
            "1910.06441",
            "2210.09246",
            "2111.03188",
            "1811.07007",
            "2303.15347",
            "2210.04580",
            "1909.06032",
            "2107.13138",
        ],
        inputs=id_input,
    )
    gr.Markdown("### Article Abstract:")
    article_abstract = gr.HTML()

    # with gr.Tab("Predict by title"):
    #     with gr.Row():
    #         title_input = gr.Textbox(label="Input title")
    #         title_title = gr.Textbox(label="Title of closest match")
    #         title_predict = gr.Textbox(label="Predicted tags")
    #         title_true = gr.Textbox(label="True tags")
    #     title_button = gr.Button("Predict")
    #     gr.Examples(
    #         examples=[
    #             "Attention is all you need",
    #             "Etale cohomology of diamonds",
    #             "Stochastic Kahler geometry from random zeros to random metrics",
    #             "Scaling asymptotics for Szego kernels on Grauert tubes",
    #             "The Wave Trace and Birkhoff Billiards",
    #         ],
    #         inputs=title_input,
    #     )

    id_button.click(
        outputs_from_id,
        inputs=[id_input, threshold_probability],
        outputs=[id_title, id_predict, id_true, article_abstract],
    )
    # title_button.click(
    #     outputs_from_title,
    #     inputs=[title_input, threshold_probability],
    #     outputs=[title_title, title_predict, title_true],
    # )

demo.launch(inbrowser=True)

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
    subject_tags = ", ".join(
        sorted(
            [subject_dict[tag] for tag in raw_categories if tag in subject_dict.keys()]
        )
    )

    return (title, subject_tags)


def parse_title(input_title):
    search = arxiv.Search(
        query=f"ti:%22{input_title}%22",
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
        max_results=1,
    )
    result = next(search.results())
    raw_categories = result.categories
    title = result.title

    with open("./data/arxiv-label-dict.json", "r") as file:
        subject_dict = json.loads(file.read())

    subject_tags = ", ".join(
        sorted(
            [subject_dict[tag] for tag in raw_categories if tag in subject_dict.keys()]
        )
    )

    return (title, subject_tags)


def outputs_from_id(input_id):
    title, true_tags = parse_id(input_id)
    predicted_tags = predict_from_text(title)

    return title, predicted_tags, true_tags


def outputs_from_title(input_title):
    title, true_tags = parse_title(input_title)
    predicted_tags = predict_from_text(title)

    return title, predicted_tags, true_tags


with gr.Blocks() as demo:
    gr.Markdown(
        "Predict the arXiv subject tags of a math article by its title using this demo."
    )
    with gr.Tab("Predict by id"):
        with gr.Row():
            id_input = gr.Textbox(label="Input ID")
            id_title = gr.Textbox(label="Title")
            id_predict = gr.Textbox(label="Predicted tags")
            id_true = gr.Textbox(label="True tags")
        id_button = gr.Button("Predict")

        gr.Examples(
            examples=[
                "1706.03762",
                "1709.07343",
                "2303.11559",
                "2107.05105",
                "1910.06441",
            ],
            inputs=id_input,
        )

    with gr.Tab("Predict by title"):
        with gr.Row():
            title_input = gr.Textbox(label="Input title")
            title_title = gr.Textbox(label="Title of closest match")
            title_predict = gr.Textbox(label="Predicted tags")
            title_true = gr.Textbox(label="True tags")
        title_button = gr.Button("Predict")
        gr.Examples(
            examples=[
                "Attention is all you need",
                "Etale cohomology of diamonds",
                "Stochastic Kähler geometry: from random zeros to random metrics",
                "Scaling asymptotics for Szegő kernels on Grauert tubes",
                "The Wave Trace and Birkhoff Billiards",
            ],
            inputs=title_input,
        )

    id_button.click(
        outputs_from_id, inputs=id_input, outputs=[id_title, id_predict, id_true]
    )
    title_button.click(
        outputs_from_title,
        inputs=title_input,
        outputs=[title_title, title_predict, title_true],
    )

demo.launch(inbrowser=True)

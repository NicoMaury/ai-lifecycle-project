from app_functions import *
import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import base64
import io
from PIL import Image
import tempfile

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MORPH])

left_card_single_image = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Original image", className="left_card", style={'textAlign': 'center'}),
                dbc.CardImg(children=None, top=True, id="original_image", style={'textAlign': 'center', 'borderRadius': '5px', 'height': '300px', 'width':'auto', 'textAlign': 'center'}),
                html.P(
                    "You should upload an image.",
                    className="card-text",
                    id="original_image_text",
                ),
            dbc.Button("Detect Animals !", color="primary", id="left_card_button"),
            ]
        ),
    ],
    style={"width": "auto", "textAlign": "center", 'background' : '#88E788', 'height': '500px'},
)

right_card_single_image = dbc.Card(
    [ 
        dbc.CardBody(
            [
                html.H4("Cropped image", className="right_card", style={'textAlign': 'center'}),
                dbc.CardImg(children=None, top=True, id="cropped_image", style={'textAlign': 'center', 'borderRadius': '5px', 'height': '300px', 'width': 'auto', 'textAlign': 'center'}),
                html.Br(),
                html.Br(),
                html.Br(),
                html.P(
                    "First upload an image to see the cropped image.",
                    className="card-text",
                    id="cropped_image_text",
                ),
                html.Br(),
            ],
            style={'textAlign': 'center'}
        ),
    ],
    style={"width": "auto", 'background' : '#88E788', 'height': '500px'},
)

left_card_video = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Original video", className="left_card", style={'textAlign': 'center'}),
                html.Br(),
                dbc.CardImg(children=None, top=True, id="original_video", style={'textAlign': 'center', 'borderRadius': '5px', 'height': '300px', 'width':'auto', 'textAlign': 'center'}),
                html.Br(),
                html.Br(),
                html.P(
                    "You should upload a video.",
                    className="card-text",
                    id="original_video_text",
                ),
            dbc.Button("Detect Animals !", color="primary", id="left_card_button_video"),
            ]
        ),
    ],
    style={"width": "auto", "textAlign": "center", 'background' : '#88E788', 'height': '600px'},
)

right_card_video = dbc.Card(
    [ 
        dbc.CardBody(
            [
                html.H4("Detected labels", className="right_card", style={'textAlign': 'center'}),
                html.P(
                    "First upload a video to see the detected labels.",
                    className="card-text",
                    id="detected_labels_text",
                ),
                dcc.Graph(id='detected_labels', figure={}),
            ],
            style={'textAlign': 'center'}
        ),
    ],
    style={"width": "auto", 'background' : '#88E788', 'height': '600px'},
)


# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row([
        html.Br(),
        dbc.Col(html.H1("Camera Trap Image Classifier", style={'textAlign': 'center', 'background' : '#88E788'})),
        html.Br(),
    ], justify='center'),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select an Image')
                ]),
                style={
                    'width': '30%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': 'auto'
                },
            ),
        ],),
    ], justify='center'),
    html.Br(),
    dbc.Col([
        dbc.Row([
            dbc.Col(left_card_single_image, width=6),
            dbc.Col(right_card_single_image, width=6),
        ]),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup(
                [dbc.Button(label, id=f"btn-{label}", color="primary", style={'color':'green'}) for label in all_labels],
                id='button_group_species',
                size="sm",  # Small, medium, or large
                className="d-flex flex-wrap center-content-center"
            ),
            html.Br(),
            dbc.ButtonGroup(
                [dbc.Input(placeholder='Add a species', id="add_species", type='text'),
                dbc.Button("Add", id="add_button", color="success")],
                size="sm",  # Small, medium, or large
            ),
        ],
        style={'textAlign': 'center'}),
    ]),
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-video',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a Video')
                ]),
                style={
                    'width': '30%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': 'auto'
                },
            ),
        ],),
    ], justify='center'),
    html.Br(),
    dbc.Col([
        dbc.Row([
            dbc.Col(left_card_video, width=6),
            dbc.Col(right_card_video, width=6),
        ]),
    ]),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    ],
    fluid=True,
    style={'background': '#D4F6D4'}
)

@app.callback(
    Output("original_image", "src"),
    Output("original_image_text", "children"),
    Input("upload-image", "contents")
)
def update_original_image(contents):
    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        image.save("original_image.jpg")
        return contents, "Press the button to detect animals."
    return None, "You should upload an image."

@app.callback(
    Output("cropped_image", "src"),
    Output("cropped_image_text", "children"),
    Input("upload-image", "contents"),
    Input("left_card_button", "n_clicks"),
    State("upload-image", "contents")
)
def update_cropped_image(contents, n_clicks, state_contents):
    if contents is not None and n_clicks is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        
        # Crop the image using MegaDetector
        cropped_image = crop_image_with_megadetector(image, detector_model)
        
        if cropped_image:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cropped_image.save(temp_file.name)
            encoded_cropped_image = base64.b64encode(temp_file.read()).decode('utf-8')
            temp_file.close()

            # Predict the label
            predicted_label, prob = predict_image_label(cropped_image, model, processor, active_labels)

            return f"data:image/jpeg;base64,{encoded_cropped_image}", f"This is a camera trap image of a {predicted_label} ({prob*100:.2f}%)."
        else:
            return image, "No animals detected in the image."
    return None, "First upload an image to see the cropped image."


@app.callback(
    Output("add_species", "value"),
    Output('button_group_species', 'children'),
    Input("add_button", "n_clicks"),
    Input("add_species", "value"),
    Input('button_group_species', 'children'),
)
def add_species(n_clicks, value, species_buttons):
    if value is None or value == " ": 
        return value, species_buttons
    
    button = ctx.triggered_id
    if button == 'add_button' and n_clicks > 0:
        global all_labels
        global active_labels
        if value in all_labels:
            return "", species_buttons
        all_labels.append(value)
        active_labels.append(value)
        global number_of_labels
        number_of_labels += 1
        return "", species_buttons + [dbc.Button(value, id=f"btn-{value}", color="primary", style={'color':'green'})]
    return value, species_buttons

@app.callback(
    Output('button_group_species', 'children', allow_duplicate=True),
    *(Input("btn-" + label, "n_clicks") for label in all_labels),
    *(Input("btn-" + label, "children") for label in all_labels),
    prevent_initial_call=True
)
def labels_checklist(*args):
    global all_labels
    global number_of_labels

    if len(args)//2 != number_of_labels:
        labels = args[(number_of_labels - 1):]
        n_clicks = args[:(number_of_labels - 1)]
    else:
        labels = args[number_of_labels:]
        n_clicks = args[:number_of_labels]

    if len(labels) != number_of_labels:
        labels = list(labels)
        labels.append(all_labels[-1])
        labels = tuple(labels)
    
    if len(n_clicks) != number_of_labels:
        n_clicks = list(n_clicks)
        n_clicks.append(None)
        n_clicks = tuple(n_clicks)

    global active_labels
    active_labels = []

    buttons_group = []

    for i in range(number_of_labels):
        if n_clicks[i] is None or n_clicks[i] % 2 == 0:
            active_labels.append(labels[i])
            buttons_group.append(dbc.Button(labels[i], id=f"btn-{labels[i]}", color="primary", style={'color':'green'}))
        else:
            buttons_group.append(dbc.Button(labels[i], id=f"btn-{labels[i]}", color="secondary", style={'color':'red'}, n_clicks=1))
    
    return buttons_group

@app.callback(
    Output("original_video", "src"),
    Output("original_video_text", "children"),
    Input("upload-video", "contents")
)
def update_original_video(contents):
    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        video = io.BytesIO(decoded)
        return contents, "Press the button to detect animals."
    return None, "You should upload a video."


@app.callback(
    Output("detected_labels", "figure"),
    Output("detected_labels_text", "children"),
    Input("original_video", "src"),
    Input("left_card_button_video", "n_clicks"),
    State("upload-video", "contents")
)
def update_detected_labels(contents, n_clicks, state_contents):
    if contents is not None and n_clicks is not None:
        video_bytes = base64.b64decode(state_contents.split(",")[1])
        video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(video_path, 'wb') as f:
            f.write(video_bytes)
        video = cv2.VideoCapture(video_path)
        
        # Process the video
        detected_labels = process_video(video, detector_model, model, processor, active_labels)
        
        # Plot the detected labels
        
        fig = plot_detected_labels(detected_labels)
        fig.update_layout(paper_bgcolor='#D4F6D4', font_color='black')
        
        return fig, "Detected labels in the video."
    fig = px.bar(None)
    fig.update_layout(paper_bgcolor='#88E788', font_color='black')
    fig.__format__
    return fig, "First upload a video to see the detected labels."


if __name__ == '__main__':
    app.run(debug=True)
dash.register_page(__name__)

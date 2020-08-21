import os
import time

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import State, Input, Output
from dash.exceptions import PreventUpdate
from sklearn.preprocessing import MinMaxScaler

_log = None

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


###############
### LAYOUTS ###
###############

def get_file_layout():
    return html.Div(
        [
            'File Path',
            dbc.Input(
                id='file_path',
                placeholder='Enter file path...',
                value='./',
                debounce=True,
                style={'width': '100%', }
            ),
        ]
    ), html.Div(
        [
            'File Name',
            dcc.Dropdown(
                id='file_name',
            ),
        ],
        style={'margin-top': '10px', }
    ), dbc.Button(
        'Refresh',
        id='refresh_btn',
        block=True,
        style={'margin-top': '0px', }
    )


def get_log_layout():
    return dbc.Row(
        [
            dbc.Col(html.Div('Episode'), width=3),
            dbc.Col(html.Label('1234 / 4321', id='episode', style={'width': '100%', 'text-align': 'right', }))
        ]
    ), dbc.Row(
        [
            dbc.Col(html.Div('Epoch'), width=3),
            dbc.Col(html.Label('12 / 43', id='epoch', style={'width': '100%', 'text-align': 'right', }))
        ]
    ), dbc.Row(
        [
            dbc.Col(html.Div('Duration'), width=3),
            dbc.Col(html.Label('12:34:56', id='duration', style={'width': '100%', 'text-align': 'right', }))
        ]
    ), dbc.Row(
        [
            dbc.Col(html.Div('Test'), width=3),
            dbc.Col(html.Label('False', id='test', style={'width': '100%', 'text-align': 'right', }))
        ]
    ),


def get_control_layout():
    search_opts = [
        ('Episode', 'episode'),
        ('Epoch', 'epoch'),
        ('Exploration', 'explored'),
        ('Mean', 'node_mean'),
        ('Success', 'success'),
    ]

    return html.Div(
        [
            'Episode',
            dbc.Input(
                id='epi_input',
                type='number',
                value=None,
                style={'width': '100%', 'text-align': 'right', },
            ),
        ],
        style={'margin-top': '10px', }
    ), dbc.ButtonGroup(
        [
            dbc.Button("|<<<", id='head_btn', n_clicks_timestamp=-1),
            dbc.Button("<<", id='m100_btn', n_clicks_timestamp=-1),
            dbc.Button("<", id='m10_btn', n_clicks_timestamp=-1),
            dbc.Button(">", id='p10_btn', n_clicks_timestamp=-1),
            dbc.Button(">>", id='p100_btn', n_clicks_timestamp=-1),
            dbc.Button(">>>|", id='tail_btn', n_clicks_timestamp=-1),
        ],
        style={'width': '100%', 'margin-top': '0px', }
    ), html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Label('Max Score')),
                    dbc.Col(dbc.Input(id='max_score', type='number', style={'text-align': 'right'})),
                ],
                justify='between',
                style={'margin-top': '20px', }
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Label('Min Score')),
                    dbc.Col(dbc.Input(id='min_score', type='number', style={'text-align': 'right'})),
                ],
                justify='between',
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Label('Window Width')),
                    dbc.Col(dbc.Input(id='window', type='number', value=100, style={'text-align': 'right'})),
                ],
                justify='between',
            ),
        ],
        id='score_control',
        hidden=False,
    ), html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(html.Label('X-Axis')),
                    dbc.Col(dcc.Dropdown(
                        id='x_axis',
                        value='0',
                    )),
                ],
                justify='between',
                style={'margin-top': '20px', }
            ),
            dbc.Row(
                [
                    dbc.Col(html.Label('Y-Axis')),
                    dbc.Col(dcc.Dropdown(
                        id='y_axis',
                        value='1',
                    )),
                ],
                justify='between',
            ),
            dbc.Row(
                [
                    dbc.Col(html.Label('Z-Axis')),
                    dbc.Col(dcc.Dropdown(
                        id='z_axis',
                        value='2',
                    )),
                ],
                justify='between',
            ),
            dbc.Row(
                [
                    dbc.Col(html.Label('Color')),
                    dbc.Col(dcc.Dropdown(
                        id='color',
                        options=[{'label': label, 'value': value} for label, value in search_opts],
                        value='node_mean',
                        clearable=False,
                    )),
                ],
                justify='between',
                style={'margin-top': '20px', },
            ),
            dbc.Row(
                [
                    dbc.Col(html.Label('Size')),
                    dbc.Col(dcc.Dropdown(
                        id='size',
                        options=[{'label': label, 'value': value} for label, value in search_opts],
                    )),
                ],
                justify='between',
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Button("Show Edges", id='edge_btn', n_clicks=0, block=True)),
                    dbc.Col(dbc.Button("Show Seeds", id='seed_btn', n_clicks=0, block=True)),
                ],
                justify='between',
                style={'margin-top': '20px', }
            ),
        ],
        id='search_control',
        hidden=True,
    ),


def get_score_layout():
    return html.Div(
        [
            dcc.Graph(
                id='score',
                style={
                    'height': '35vh',
                },
            ),
            dcc.Graph(
                id='live_score',
                style={'height': '55vh', },
            ),
        ],
    )


def get_search_layout():
    return html.Div(
        children=dcc.Graph(
            id='search',
            style={'height': '90vh', },
        ),
    )


def get_layout():
    return html.Div(
        [
            dcc.Store(
                id='data',
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody([html.H4('File', className='card-title'), *get_file_layout()]),
                            ),
                            dbc.Card(
                                dbc.CardBody([html.H4('Log', className='card-title'), *get_log_layout()]),
                                style={'margin-top': '5px', }
                            ),
                            dbc.Card(
                                dbc.CardBody([html.H4('Controls', className='card-title'), *get_control_layout()]),
                                style={'margin-top': '5px', }
                            ),
                        ],
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                dbc.Tabs(
                                    [
                                        dbc.Tab(get_score_layout(), label='Score Graph', tab_id='score', ),
                                        dbc.Tab(get_search_layout(), label='Search Graph', tab_id='search', ),
                                    ],
                                    id='graphs',
                                    style={'cursor': 'pointer', },
                                ),
                            ),
                        ),
                        width=9,
                    ),
                ]
            )
        ],
        style={
            'height': '120vh',
            'background': 'Gainsboro',
            'padding': '5px 15px',
        }
    )


app.layout = get_layout()


########################
### HELPER FUNCTIONS ###
########################

def get_df(epi, win=None):
    if win is None:
        win = epi

    epi = max(0, min(_log.shape[0], epi + 1))

    head = max(0, epi - win - 1)

    return _log.iloc[head:epi]


def get_log(episode=None):
    episode = _log.shape[0] - 1 if episode is None else episode
    epoch = _log.epoch.iloc[-1] if episode is None else _log.epoch.iloc[episode]
    epoch += 1

    ts = _log.time.iloc[-1] if episode is None else _log.time.iloc[episode]
    duration = time.strftime('%X', time.gmtime(ts - _log.time.iloc[0]))

    test = _log.test.iloc[-1] if episode is None else _log.test.iloc[episode]

    return episode, epoch, duration, test


def get_score_figure(episode=None, window=None):
    if episode is None:
        episode = _log.shape[0] - 1

    df = get_df(episode, window)

    episode = df.episode

    values = [
        ['node_score', 'Node Score', 'steelblue', 1, 'solid'],
        ['seed_mean', 'Seed Mean', 'olivedrab', 4, 'solid'],
        ['best_mean', 'Best Mean', 'crimson', 4, 'dash'],
        ['best_ci_low', 'Best CI_Low', 'tomato', 4, 'dot'],
    ]

    fig = go.Figure()

    for val, name, color, width, style in values:
        fig.add_trace(go.Scatter(
            x=episode,
            y=df[val],
            mode='lines',
            name=name,
            line={'color': color, 'width': width, 'dash': style},
        ))

    epochs = df.groupby('epoch')['episode'].transform(max).unique()

    shapes = []
    for epoch in epochs:
        shape = dict(
            type='line',
            yref='paper', y0=0, y1=1,
            xref='x', x0=epoch, x1=epoch,
            line={'color': 'gold', 'width': 4, 'dash': 'dash'},
        )

        shapes.append(shape)

    fig.update_layout(
        xaxis=dict(
            showline=True,
            zeroline=False,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            gridcolor='rgb(204, 204, 204)',
            gridwidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=0,
        ),
        shapes=shapes,
        plot_bgcolor='white',
    )

    return fig


def get_search_df(df):
    # mark seeds
    df['is_seed'] = df['node_id'].isin(df['seed_id'])

    # df['created'] = df.groupby('node_id')['episode'].transform(min)
    # for each node episode value represents the episode when it was created
    # also seed node is the first seed_id encountered (testing may set the seed_id to itself)
    tmp = df[df.groupby('node_id')['episode'].transform(min) == df.episode]
    tmp = tmp.reset_index(drop=True)

    # for later calculations only last episode of each node will be used
    df = df[df.groupby('node_id')['episode'].transform(max) == df.episode]
    df = df.reset_index(drop=True)

    df[['episode', 'seed_id', 'seed_mean']] = tmp[['episode', 'seed_id', 'seed_mean']]

    # increase epoch value to represent nth epoch
    df['epoch'] = df['epoch'] + 1

    # calculate explored value for each node
    # for each seed count the number of children
    tmp = df.groupby('seed_id')['node_id'].nunique()
    tmp = tmp.to_frame('explored')
    # for each node assign the explored value
    df = df.merge(tmp, left_on='node_id', right_index=True, how='outer')
    # for non seeds set exploration to 0
    df['explored'] = df['explored'].fillna(0)

    # success is the difference of node.mean and seed.mean
    df['success'] = df['node_mean'] - df['seed_mean']
    # for nodes without seeds (first node of each epoch) set success to 0
    df['success'] = df['success'].fillna(0)

    return df


def get_node_graph(df, color, magnitude):
    # text for hover label
    text = df[['episode', 'node_id', 'node_mean', 'explored', 'success', 'epoch']]

    mag = 12

    if magnitude is not None and not df[magnitude].empty:
        # map magnitude to positive numbers
        scalar = MinMaxScaler()

        m = df[magnitude].values.reshape(-1, 1)

        m = scalar.fit_transform(m)
        m = m.flatten()

        # emphasize on bigger values for better visualization
        mag = (np.exp(m * 2.8) + 1) * 7

    return go.Scatter3d(
        x=df.x,
        y=df.y,
        z=df.z,
        mode='markers',
        marker=dict(
            color=df[color],
            size=mag,
            showscale=True,
        ),
        text=text,
        hovertemplate=
        '<b>Episode</b>: %{text[0]}<br>' +
        '<b>Id</b>: %{text[1]}<br>' +
        '<b>Mean</b>: %{text[2]:.2f}<br>' +
        '<b>Explored</b>: %{text[3]}<br>' +
        '<b>Success</b>: %{text[4]:.2f}<br>' +
        '<b>Epoch</b>: %{text[5]}<br><br>' +
        '<b>X</b>: %{x:.2f}<br>' +
        '<b>Y</b>: %{y:.2f}<br>' +
        '<b>Z</b>: %{z:.2f}<br>' +
        "<extra></extra>",
        showlegend=False,
    )


def get_edge_graphs(df):
    # get seed coordinates
    tmp = df[df.is_seed][['node_id', 'x', 'y', 'z']]
    tmp = tmp.reset_index(drop=True)
    df = df.merge(tmp, left_on='seed_id', right_on='node_id', how='outer', suffixes=['_node', '_seed'])

    df['gap'] = None

    # separate seeds
    seed = df[df.is_seed.astype(bool)].reset_index(drop=True)
    node = df[~df.is_seed.astype(bool)].reset_index(drop=True)

    # set edge values
    values = [
        [node, 3, 'crimson'],
        [seed, 7, 'olivedrab'],
    ]

    # get edges
    edges = []
    for dff, width, color in values:
        x = dff[['x_seed', 'x_node', 'gap']].values.flatten()
        y = dff[['y_seed', 'y_node', 'gap']].values.flatten()
        z = dff[['z_seed', 'z_node', 'gap']].values.flatten()

        edge = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(
                width=width,
                color=color,
            ),
            hoverinfo='none',
            showlegend=False,
        )

        edges.append(edge)

    return edges


#################
### CALLBACKS ###
#################

@app.callback(
    Output('file_name', 'options'),
    [
        Input('file_path', 'value'),
        Input('refresh_btn', 'n_clicks'),
    ]
)
def update_file_path(path, click):
    files = sorted([file for file in os.listdir(path) if file.endswith('.log')])
    return [{'label': file, 'value': os.path.join(path, file)} for file in files]


@app.callback(
    Output('data', 'data'),
    [
        Input('file_name', 'value'),
        Input('refresh_btn', 'n_clicks'),
    ]
)
def update_data(name, click):
    global _log

    if name is None:
        _log = None
        return {}

    _log = pd.read_csv(name)
    _log['episode'] = _log.index

    episode, epoch, duration, test = get_log()

    data = {
        'episode': episode,
        'epoch': epoch,
        'duration': duration,
        'test': test,
    }

    return data


@app.callback(
    Output('epi_input', 'value'),
    [
        Input('data', 'modified_timestamp'),
        Input('head_btn', 'n_clicks_timestamp'),
        Input('m100_btn', 'n_clicks_timestamp'),
        Input('m10_btn', 'n_clicks_timestamp'),
        Input('p10_btn', 'n_clicks_timestamp'),
        Input('p100_btn', 'n_clicks_timestamp'),
        Input('tail_btn', 'n_clicks_timestamp'),
    ], [
        State('epi_input', 'value'),
        State('data', 'data'),
    ]
)
def update_episode(dts, head, m100, m10, p10, p100, tail, epi, data):
    if _log is None:
        return None

    buttons = [head, m100, m10, p10, p100, tail]
    btn_max = max(buttons)

    if dts > btn_max:
        return data['episode']

    btn_values = [0, epi - 100, epi - 10, epi + 10, epi + 100, data['episode']]

    btn = np.argmax(buttons)
    epi = btn_values[btn]
    epi = max(0, min(data['episode'], epi))

    return epi


@app.callback(
    [
        Output('episode', 'children'),
        Output('epoch', 'children'),
        Output('duration', 'children'),
        Output('test', 'children'),
    ], [
        Input('epi_input', 'value'),
    ], [
        State('data', 'data'),
    ]
)
def update_log(epi, data):
    if _log is None:
        return None, None, None, None

    epi = max(0, min(data['episode'], epi))

    episode, epoch, duration, test = get_log(epi)

    episode = f'{episode} / {data["episode"]}'
    epoch = f'{epoch} / {data["epoch"]}'
    duration = f'{duration} / {data["duration"]}'
    test = f'{test}'

    return episode, epoch, duration, test


@app.callback(
    [
        Output('score_control', 'hidden'),
        Output('search_control', 'hidden'),
    ], [
        Input('graphs', 'active_tab'),
    ]
)
def update_controls(tab):
    p = tab == 'search'
    return p, not p


@app.callback([
    Output('x_axis', 'options'),
    Output('y_axis', 'options'),
    Output('z_axis', 'options'),
], [
    Input('epi_input', 'value'),
]
)
def update_axis_options(epi):
    opts = []

    if _log is None or epi is None:
        return [opts] * 3

    df = get_df(epi)

    if df is not None:
        attrs = df.columns
        attrs = attrs[attrs.str.isdecimal()]
        opts = [(f'Attribute {attr}', attr) for attr in attrs]
        opts = [{'label': label, 'value': value} for label, value in opts]

    return opts, opts, opts


@app.callback(
    Output('score', 'figure'),
    [
        Input('data', 'modified_timestamp')
    ]
)
def update_score(dts):
    if _log is None:
        return {}

    return get_score_figure()


@app.callback(
    Output('live_score', 'figure'),
    [
        Input('epi_input', 'value'),
        Input('max_score', 'value'),
        Input('min_score', 'value'),
        Input('window', 'value'),
        Input('graphs', 'active_tab'),
    ]
)
def update_live_score(epi, max_score, min_score, window, tab):
    if tab != 'score':
        raise PreventUpdate

    if _log is None or epi is None:
        return {}

    fig = get_score_figure(epi, window)

    df = get_df(epi, window)

    df_max_score = df['node_score'].max()
    df_min_score = df['node_score'].min()

    max_score = df_max_score + abs(df_max_score) * .1 if max_score is None else max_score
    min_score = df_min_score - abs(df_min_score) * .1 if min_score is None else min_score
    if min_score >= max_score:
        min_score = max_score - 100

    fig.update_yaxes(range=[min_score, max_score])

    return fig


@app.callback(
    Output('search', 'figure'),
    [
        Input('epi_input', 'value'),
        Input('x_axis', 'value'),
        Input('y_axis', 'value'),
        Input('z_axis', 'value'),
        Input('color', 'value'),
        Input('size', 'value'),
        Input('edge_btn', 'n_clicks'),
        Input('seed_btn', 'n_clicks'),
        Input('graphs', 'active_tab'),
    ]
)
def update_search(epi, x, y, z, color, magnitude, edge, seed, tab):
    if tab != 'search':
        raise PreventUpdate

    if _log is None or epi is None:
        return {}

    if not all([x, y, z]):
        return {}

    df = get_df(epi)

    df = df.rename(columns={x: 'x', y: 'y', z: 'z'})

    df = get_search_df(df)

    if seed % 2 != 0:
        df = df[df.is_seed]

    data = [get_node_graph(df, color, magnitude)]

    if edge % 2 != 0:
        data.extend(get_edge_graphs(df))

    layout = go.Layout(
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=0,
        ),
        scene=go.layout.Scene(
            xaxis=go.layout.scene.XAxis(showspikes=False),
            yaxis=go.layout.scene.YAxis(showspikes=False),
            zaxis=go.layout.scene.ZAxis(showspikes=False),
        ),
        hoverlabel=dict(
            bgcolor='whitesmoke',
        ),
    )

    return go.Figure(data=data, layout=layout)


@app.callback(
    [
        Output('edge_btn', 'children'),
        Output('edge_btn', 'outline'),
        Output('seed_btn', 'children'),
        Output('seed_btn', 'outline'),
    ], [
        Input('edge_btn', 'n_clicks'),
        Input('seed_btn', 'n_clicks')
    ]
)
def update_search_buttons(edge, seed):
    edge_vals = [
        ['Show Edges', False],
        ['Hide Edges', True],
    ]

    seed_vals = [
        ['Show Seeds', False],
        ['Show All', True],
    ]

    return edge_vals[edge % 2] + seed_vals[seed % 2]


if __name__ == '__main__':
    app.run_server(debug=True)

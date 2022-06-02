import datetime
import os
import warnings
from collections import OrderedDict

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pathlib

from datahelper import *
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# data_files = os.listdir(os.path.join('data'))
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
data_files = os.listdir(DATA_PATH)

#### App layout
app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container([
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H1("Daily Analytics Dashboard",
                            className="page-title",
                            style={'textAlign': 'center',
                                   'fontWeight': 'bold'})
                ])
            ], color='primary', inverse=True)
        ])
    ]),
    html.Br(),
    dcc.Dropdown(id='input_file',
                 options=[{'label': x, 'value': x} for x in sorted(data_files)],
                 style={"width": "100%"},
                 placeholder='',
                 searchable=True,
                 clearable=True),
    html.Br(),
    # dcc.Store(id='datafile'),
    html.Br(),
    html.Div(id='info')
])


################################################################
@app.callback(Output('info', 'children'),
              Input('input_file', 'value'))
def update_data(input_file):
    if input_file:
        # filepath = 'data/' + str(input_file)
        filepath = DATA_PATH.joinpath(input_file)
        df = cleanup(filepath)
        fname = df.Filename.iloc[0].split('\\')[-1]
        initiated = df['Start Datetime'].min()
        terminated = df['End Datetime'].max()
        info = OrderedDict([
            ("Attribute", ["Filename", "Duration"]),
            ("Description", [fname, "{} to {}".format(initiated, terminated)])
        ])
        df_info = pd.DataFrame(info)
        print(df)
        # Processing for stats section
        df_time = df.copy()
        cols = df_time.columns.tolist()
        cutoff = cols.index('Active 1')
        time_begin = df_time['Start Datetime'].min().time()
        time_to_minus = datetime.timedelta(hours=time_begin.hour,
                                           minutes=time_begin.minute,
                                           seconds=time_begin.second)

        def transform_to_time(row):
            start_time = row['Start Datetime']
            for col in cols[cutoff:]:
                secs = row[col]
                if secs != 0:
                    row[col] = start_time + datetime.timedelta(seconds=secs) - time_to_minus
                else:
                    pass
            return row

        df_time = df_time.apply(transform_to_time, axis=1)
        numSubjects = len(df_time)
        stats = []
        for i in range(numSubjects):
            test_row = df_time.iloc[i]
            for col in cols[cutoff:]:
                stat = []
                if (type(test_row[col]) != float) and (type(test_row[col]) != int):
                    stat.append(test_row['Subject'])
                    stat.append(test_row[col])
                    stat.append(test_row[col] + datetime.timedelta(seconds=15))
                    stat.append(col.split(' ')[0])
                    stats.append(stat)
                else:
                    pass

        timeline_data = pd.DataFrame(stats, columns=['Subject', 'Start', 'End', 'Type'])
        timeline_graph = px.timeline(timeline_data,
                                     x_start="Start",
                                     x_end="End",
                                     y="Subject",
                                     color='Type',
                                     title='Actions Timeline')

        timeline_graph.update_xaxes(tickformat="%H:%M:%S",
                                    tickformatstops=[dict(dtickrange=[1800000, 10800000], value="%H:%M")])

        timeline_graph.update_yaxes(categoryorder="category ascending")

        timeline_graph.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        # Reward histogram
        df_filtered = filtered_reward(df)
        reward_data = []
        for s in df_filtered['Subject']:
            cl = df_filtered[df_filtered['Subject'] == s]['cleanedIntervals'].values[0]
            for val in cl:
                reward_data.append([s, val])
        df_dist = pd.DataFrame(reward_data, columns=['ID', 'Second'])
        numBins = round(max(df_dist['Second']) / 20)

        reward_histogram = px.histogram(df_dist,
                                        x="Second",
                                        color="ID",
                                        nbins=numBins,
                                        # marginal="box", # can be `box`, `violin`
                                        hover_data=df_dist.columns,
                                        color_discrete_sequence=px.colors.qualitative.Dark24,
                                        title='Inter-Reward Time Interval Distribution')

        # reward_histogram.update_layout(autosize=False,
        #                 width=1100,
        #                 height=700,
        #                 margin=dict(l=50,
        #                             r=50,
        #                             b=100,
        #                             t=100,
        #                             pad=4))

        # Processing for table section
        subjects, n_active, n_inactive, n_reward, n_timeout, latencies = [], [], [], [], [], []

        for subject in sorted(df['Subject']):
            dff = df[df['Subject'] == subject]
            subjects.append(subject)
            n_active.append(dff['Active Lever Presses'].values[0])
            n_inactive.append(dff['Inactive Lever Presses'].values[0])
            n_reward.append(dff['Reward'].values[0])
            n_timeout.append(dff['Timeout'].values[0])
            latencies.append(dff['Reward 1'].values[0])

        actions = OrderedDict([
            ("Subject", subjects),
            ("Active", n_active),
            ("Inactive", n_inactive),
            ("Reward", n_reward),
            ("Timeout", n_timeout),
            ("Latency (seconds)", latencies)
        ])
        df_actions = pd.DataFrame(actions)
        print(df_actions)
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dash_table.DataTable(
                            data=df_info.to_dict('records'),
                            columns=[{'id': c, 'name': c} for c in df_info.columns],
                            style_header={
                                'backgroundColor': 'rgb(99, 111, 112)',
                                "fontWeight": "bold",
                                'color': 'white'
                            },
                            style_data={
                                'backgroundColor': 'rgba(255, 0, 0, 0)',
                                'color': 'white'
                            },
                            style_cell={'fontSize': 18,
                                        'fontFamily': 'verdana',
                                        'textAlign': 'right'
                                        },
                            style_cell_conditional=[
                                {
                                    'if': {'column_id': c},
                                    'textAlign': 'left'
                                } for c in ['Info']
                            ],
                        ),
                    ], color='dark', inverse=False),
                ], width=12),
            ], className='file-info'),

            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H2('Active'),
                            html.Br(),
                            html.H5('Mean: ' + str(round(df_actions['Active'].mean(), 2))),
                            html.H5('SD:  ' + str(round(df_actions['Active'].std(), 2)))
                        ], style={'textAlign': 'left'})
                    ], color='primary', inverse=True),
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H2('Inactive'),
                            html.Br(),
                            html.H5('Mean: ' + str(round(df_actions['Inactive'].mean(), 2))),
                            html.H5('SD:  ' + str(round(df_actions['Inactive'].std(), 2)))
                        ], style={'textAlign': 'left'})
                    ], color='primary', inverse=True),
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H2('Reward'),
                            html.Br(),
                            html.H5('Mean: ' + str(round(df_actions['Reward'].mean(), 2))),
                            html.H5('SD:  ' + str(round(df_actions['Reward'].std(), 2)))
                        ], style={'textAlign': 'left'})
                    ], color='primary', inverse=True),
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H2('Timeout'),
                            html.Br(),
                            html.H5('Mean: ' + str(round(df_actions['Timeout'].mean(), 2))),
                            html.H5('SD:  ' + str(round(df_actions['Timeout'].std(), 2)))
                        ], style={'textAlign': 'left'})
                    ], color='primary', inverse=True),
                ], width=3),
            ], className="overall-stats"),

            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=timeline_graph)
                        ], style={'textAlign': 'center'})
                    ]),
                ], width=12),
            ], className='timeline-graph'),

            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=reward_histogram)
                        ], style={'textAlign': 'center'})
                    ]),
                ], width=12),
            ], className='reward-distribution-graph'),

            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dash_table.DataTable(
                            data=df_actions.to_dict('records'),
                            columns=[{'id': c, 'name': c} for c in df_actions.columns],
                            style_header={
                                'backgroundColor': 'rgb(33, 125, 187)',
                                "fontWeight": "bold",
                                'color': 'white'
                            },
                            style_data={
                                'backgroundColor': 'rgba(255, 0, 0, 0)',
                                'color': 'white'
                            },
                            style_cell={'fontSize': 18,
                                        'fontFamily': 'verdana',
                                        'textAlign': 'right'
                                        },
                            style_cell_conditional=[
                                {
                                    'if': {'column_id': c},
                                    'textAlign': 'left'
                                } for c in ['Subject']
                            ],
                        ),
                    ], color='info', inverse=False),
                ], width=12),
            ], className='individual-stats'),
            html.Br(),
            html.Br(),
            html.Br(),
        ])
    else:
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Br(),
                            html.Br(),
                            html.H3("Select a file above",
                                    className="page-title",
                                    style={'textAlign': 'center'}),
                            html.Br(),
                            html.Br()
                        ])
                    ])
                ])
            ]),
            html.Br()
        ])


#### Run server ############################################################
if __name__ == "__main__":
    app.run_server(debug=True)

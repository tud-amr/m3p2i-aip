from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from m3p2i_aip.utils import path_utils
import multiprocessing
import time
import plotly.graph_objects as go

def start_dash_server():

    file_path = path_utils.get_plot_path() +'/data_battery.csv'
    np.savetxt(file_path, [100], fmt='%.1f')

    app = Dash(__name__)

    app.layout = html.Div([
        html.H1('Battery Level of Robot', style={'marginLeft':'430px'}),
        dcc.Interval(id="interval",
                    interval=500, # increase the counter `n_intervals` every 0.5 seconds
                    n_intervals=0
                    ),
        dcc.Graph(id="graph"),
    ])

    @app.callback(
        Output("graph", "figure"), 
        Input("interval", "n_intervals"))
    def display_graph(n_intervals):
        df1 = pd.read_csv(path_utils.get_plot_path() +'/data_battery.csv')
        bettery_level = float(df1.columns[0])
        if bettery_level > 80:
            color = "green"
        elif bettery_level > 60:
            color = "blue"
        else:
            color = "red"
        fig = go.Figure()
        fig.add_bar(y=[0, df1.columns[0], 0], width=0.5, name="Robot", marker_color=color)
        fig.update_traces(texttemplate = df1.columns[0], textposition = 'inside')
        # fig.update_xaxes(showticklabels=False)
        # fig.update_xaxes(type='category')
        fig.update_xaxes(visible=False)
        fig.update_yaxes(range=[0, 100])
        fig.update_layout(title="",width=1200,
                          height=800,
                          xaxis_title="Robot",
                          yaxis_title="Battery level",
                          legend_title="", showlegend=True,
                          margin=dict(l=80, r=80, b=10, t=0, pad=0),)
        return fig

    def run():
        app.run_server(debug=True, port=8040)

    # Run on a separate process so that it doesn"t block
    app.server_process = multiprocessing.Process(target=run)
    app.server_process.start()

if __name__== "__main__":
    start_dash_server()
    # plot the new updated data
    file_path = path_utils.get_plot_path() +'/data_battery.csv'
    for i in range(1000):
        time.sleep(0.2)
        np.savetxt(file_path, [100-i/10], fmt='%.1f')
        # print(i)
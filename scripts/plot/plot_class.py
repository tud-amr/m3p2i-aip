from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import sys
import numpy as np
import os
sys.path.append('../')
from utils import path_utils
import multiprocessing
import time

def start_dash_server():

    file_path = path_utils.get_plot_path() +'/data_battery.csv'
    np.savetxt(file_path, [100], fmt='%.1f')

    app = Dash(__name__)

    app.layout = html.Div([
        html.H1('Battery Level of Robot'),
        dcc.Interval(id="interval",
                    interval=1000,
                    n_intervals=0
                    ),
        dcc.Graph(id="graph"),
    ])

    @app.callback(
        Output("graph", "figure"), 
        Input("interval", "n_intervals"))
    def display_graph(n_intervals):
        df1 = pd.read_csv(path_utils.get_plot_path() +'/data_battery.csv')
        # print(df1.columns[0])
        df = pd.DataFrame({
            "Robot": ["Base", "Real Robot"],
            "Battery Level": [100, df1.columns[0]],
            "Color": [100, df1.columns[0]]
        })
        fig = px.bar(df, x="Robot", y="Battery Level", color="Color", width=1200, height = 800) 
        fig.update_traces(width=0.2)

        return fig

    def run():
        app.run_server(debug=True, port=8040)

    # Run on a separate process so that it doesn"t block
    app.server_process = multiprocessing.Process(target=run)
    app.server_process.start()

if __name__== "__main__":
    start_dash_server()
    time.sleep(5)
    # plot the new updated data
    file_path = path_utils.get_plot_path() +'/data_battery.csv'
    for i in range(10000):
        np.savetxt(file_path, [100-i/100], fmt='%.1f')
        # print(i)
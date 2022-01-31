import os
# import dash_core_components as dcc
from dash import dcc
# import dash_html_components as html
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


#new comment

# Navbar, layouts, custom callbacks
from utils.navbar import Navbar
from utils.layouts import (
    eafLayout
)
import utils.callbacks

# Import app
from utils.dashapp import app

# Import server for deployment
from utils.dashapp import srv as server


app_name = os.getenv("DASH_APP_PATH", "/meltshop-dashboard")

# Layout variables, navbar, header, content, and container
nav = Navbar()

header = dbc.Row(
    dbc.Col(
        html.Div(
            [
                html.H2(children="Meltshop Dashboard"),
            ]
        )
    ),
    className="banner",
)

content = html.Div([dcc.Location(id="url"), html.Div(id="page-content")])

container = dbc.Container([header, content],fluid=True) 
# see help(dbc.Container) fluid = True, expands to fill available space.

#
# Menu callback, set and return
# Declair function  that connects other pages with content to container
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname in [app_name, app_name + "/"]:
        return html.Div(
            [
                dcc.Markdown(
                    """Home Page"""
                )
            ],
            className="home",
        )

    elif pathname.endswith("/eaf"):
        return eafLayout
    elif pathname.endswith("/lrf"):
        return "Page under construction..."
    elif pathname.endswith("/atomizing"):
        return "Page under construction..."
    else:
        return "Select an option from the Corner Menu"


# Set layout to Nav and Container
app.layout = html.Div([nav, container])


if __name__ == '__main__':
    app.run_server(debug=True)

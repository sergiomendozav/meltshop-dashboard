# import dash and bootstrap components
import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_components import themes


# set app variable with dash, set external style to bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP], # dbc.themes.SANDSTONE # https://codepen.io/chriddyp/pen/bWLwgP.css
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Meltshop Dashboard"
# set app server to variable for deployment
srv = app.server

# set app callback exceptions to true
app.config.suppress_callback_exceptions = True

# set applicaiton title
app.title = "Meltshop Data Visualization"

# Import Bootstrap from Dash
import os

import dash_bootstrap_components as dbc


app_name = os.getenv("DASH_APP_PATH", "/meltshop-dashboard")

# Navigation Bar fucntion
def Navbar():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("EAF", href=f"{app_name}/eaf")),
            dbc.NavItem(dbc.NavLink("LRF", href=f"{app_name}/lrf")),
            dbc.NavItem(dbc.NavLink("Atomizing", href=f"{app_name}/atomizing")),
        ],
        brand="Home",
        brand_href=f"{app_name}",
        sticky="top",
        color="light",
        dark=False,
        expand="lg",
    )
    return navbar

# Dash components, html, and dash tables
# import dash_core_components as dcc
from logging import disable
from dash import dcc
# import dash_html_components as html
from dash import html

# Import Bootstrap components
import dash_bootstrap_components as dbc

import utils.data as data


TapOptions = [
            {'label': 'Tap 8', 'value': 8},
            {'label': 'Tap 7', 'value': 7},
            {'label': 'Tap 6', 'value': 6},
            {'label': 'Tap 5', 'value': 5},
            {'label': 'Tap 4', 'value': 4},
            {'label': 'Tap 3', 'value': 3}
                 ]


# Functions
# -----------------------------------------------------------------------------

def drawGraph(id_):
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(id=id_, config={"displayModeBar": True, 'editable':True})
            ])
        ),  
    ])

def Row1Col(func, fig, width): 
    row =   dbc.Row([dbc.Col([func(fig)], width=width)], align='center')
    return row

def Row2Col(func1, fig1, w1, func2, fig2, w2):
    rows = dbc.Row([dbc.Col([func1(fig1)], width=w1),
                    dbc.Col([func2(fig2)], width=w2)], align='center')
    return rows

def Row3Col(func1, fig1, w1, func2, fig2, w2, func3,fig3, w3):
    rows = dbc.Row([dbc.Col([func1(fig1)], width=w1),
                    dbc.Col([func2(fig2)], width=w2),
                    dbc.Col([func3(fig3)], width=w3)], align='center')
    return rows

def RangeSlider(id_,min,max):
    div = html.Div(dcc.RangeSlider(
                        id=id_,
                        min=min,
                        max=max,
                        step=1,
                        value=[min, max]))
    return div

def Dropdown(id_,options,value):
    dd = dcc.Dropdown(
        id=id_,
        options=options,
        value=value,
        multi=True
    )
    return dd

def Checklist(id_,options,value):
    chl = dcc.Checklist(
        id=id_,
        options=options,
        value=value, #['NYC', 'MTL']
        labelStyle={'display': 'inline-block'}
)

    return chl


EAF_submenus = ['Electrical','Chemical','Slag','Commodities','Crew','PX3']



# EAF sub Layouts


# Electrical
electrical_layout = html.Div(children=[
    dbc.Card([

        dbc.CardBody([
            Row2Col(drawGraph,'g_kwhTon',6,drawGraph,'g_Ontime_PX3',6),
            html.Br(),
            Row2Col(drawGraph,'g_HeatAvgMW',6,drawGraph,'g_PrimVolts',6),
            html.Br(),
            Row2Col(drawGraph,'Electrode_adds',6,drawGraph,'Electrode_adds_1',6),
            html.Br(),
            Row1Col(drawGraph,'g_RWI',12)
        ]),

        dcc.Interval(
            id='EAF-electrical-interval',
            interval=1000*3600, # Refresh data every 1 hour.
            n_intervals=0
        )
    ]) 
])



# Chemical
chem_layout = html.Div(children=[
    dbc.Card([

        dbc.CardBody([
            Row1Col(drawGraph,'g_O2perTon',12),
            html.Br(),
            Row1Col(drawGraph,'g_PPM',12),
            html.Br(),
            Row1Col(drawGraph,'g_Temp',12),
            html.Br(),
            Row1Col(drawGraph,'g_O2scfLbC',12)
        ]), #CardBody

        dcc.Interval(
            id='EAF-chem-interval',
            interval=1000*3600, # Refresh data every 120 Sec.
            n_intervals=0
        ) #Interval
    ]) #Card 
]) #Div

# Slag
slag_layout = html.Div(children=[
    dbc.Card([

        dbc.CardBody([
            Row2Col(drawGraph,'g_B3',6,drawGraph,'g_B3_Group',6),
            html.Br(),
            Row2Col(drawGraph,'g_SlagMass',6,drawGraph,'g_Lime',6),
            html.Br(),
            Row1Col(drawGraph,'g_Slag_Tap_Ratio',12),
            html.Br(),
            Row2Col(drawGraph,'g_Cr_Partition',6,drawGraph,'g_Cr_Part_FeO',6),
            html.Br(),
            Row2Col(drawGraph,'g_CaO',6,drawGraph,'g_MgO',6),
            html.Br(),
            Row3Col(drawGraph,'g_FeO',4,
                    drawGraph,'g_Al2O3',4,
                    drawGraph,'g_SiO2',4),
            html.Br(),
            Row2Col(drawGraph,'g_Fe_Loss',6,drawGraph,'g_MgO_Loss',6),
            html.Br(),
            Row1Col(drawGraph,'g_matrix',12),
            html.Br()
        ]),

        dcc.Interval(
            id='EAF-slag-interval',
            interval=1000*3600, # Refresh data every 120 Sec.
            n_intervals=0
        )
    ]) 
])
# Commodities


# Crew

# PX3frame

PX3_layout = html.Div(children=[
    dbc.Card([

        dbc.CardBody([
            Row2Col(drawGraph,'g_Currents',6,drawGraph,'g_MaxTap',6),
            html.Br(),
            Row2Col(drawGraph,'g_MeanSF',6,drawGraph,'g_MWH1',6),
            html.Br(),
            Row3Col(drawGraph,'g_NCC1',4,
                     drawGraph,'g_NCC2',4,
                     drawGraph,'g_NCC3',4,),
            html.Br(),
            Row1Col(drawGraph,'g_I2H',12),
            # dbc.Row([dbc.Col(
            #     [Dropdown(id_='dd_tap',options=TapOptions,value=[6,7,8]),
            #     drawGraph('g_PowerCurve'), 
            #     RangeSlider('powercurveslider',data.df['HeatNo'].min(),data.df['HeatNo'].max()), 
            #     html.Div(id='powercurveslider_output')
            #     ],width=6), #Col
            #     dbc.Col(drawGraph('g_I2H'),width=6)
            #]), #Row
            html.Br()
        ]), #CardBody
        dcc.Interval(
            id='EAF-PX3-interval',
            interval=1000*3600, # Refresh data every 1 hour.
            n_intervals=0
                    ) #interval
    ]) #Card
]) #html.Div




# Main EAF Layout
eafLayout = html.Div(children=[

    html.H2(children="EAF"),

    dcc.Dropdown(
        id='EAF_dropdown_menu',
        options=[{'label': i, 'value': i} for i in EAF_submenus],
        value='Electrical'
    ),

    html.Div(id="EAF_sub_layout")
])



# example using matplotlib inside of dash
# What I learned: if the return on the decorated (callback) 
# function (read_data(n) in this case) has only one output, 
# then in the decorator @app.callback(Output(),[Input()]), 
# the Output() should not have brackets...only brackets when multiple 
# objects on the decorated function return.



# in data.py
#------------------------------------------------
# from plotly.tools import mpl_to_plotly
# import dash_core_components as dcc
# import matplotlib.pyplot as plt

# fig_plt = plt.figure()
# x = [1,2]
# y = [2,3]
# ax = fig_plt.add_subplot(111)
# ax.plot(x,y)
# ax.grid(True)
# plotly_fig = mpl_to_plotly(fig_plt)

# in callbacks.py
#------------------------------------------------
# elif value == 'Chemical':
#         return chemical_layout


# @app.callback(Output('g_fig_plt','figure'),
#             [Input('EAF-chemical-interval','n_intervals')])
# def read_data(n):
#     importlib.reload(utils.data)
#     return data.plotly_fig

# in layouts.py
#------------------------------------------------
# chemical_layout = html.Div(children=[
#     html.Div([
#         dbc.Card(
#             dbc.CardBody([
#                 dcc.Graph(id='g_fig_plt', config={"displayModeBar": True, 'editable':True})
#             ]) #CardBody
#         ), #Card  
#     ]), #Div
#         dcc.Interval(
#             id='EAF-chemical-interval',
#             interval=1000*1000, # Refresh data every 10 sec
#             n_intervals=0
#        ) #Interval
#     ]) #parent Div 
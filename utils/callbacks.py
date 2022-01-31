import utils
import plotly.express as px 
from dash.dependencies import Input, Output
# Import app
from utils.dashapp import app
# Import DATA
#from utils.data import *
import utils.data as data
import utils.frame as frame
# To reload libraries
import importlib
# Import layouts
from utils.layouts import electrical_layout, chem_layout, slag_layout, PX3_layout


# CALLBACKS
# -----------------------------------------------------------------------------
# callback for EAF Dropdown Menu 
@app.callback(Output("EAF_sub_layout", "children"),
[Input('EAF_dropdown_menu','value')])
def display_EAF_sublayout(value):
    if value == 'Electrical':
        return electrical_layout   
    elif value == 'Chemical':
        return chem_layout
    elif value == 'Slag':
        return slag_layout
    elif value == 'Commodities':
        return "sub menu under construction"    
    elif value == 'Crew':
        return "sub menu under construction"  
    elif value == 'PX3':
        return PX3_layout



# callback to update EAF Electrical Layout with new data
#--------------------------------------------------------------------------
@app.callback([Output('g_kwhTon','figure'),
            Output('g_PrimVolts','figure'),
            Output('Electrode_adds','figure'),
            Output('Electrode_adds_1','figure'),
            Output('g_HeatAvgMW','figure'),
            Output('g_Ontime_PX3','figure'),
            Output('g_RWI','figure')
            ],
            [Input('EAF-electrical-interval','n_intervals')])
def read_data(n):

    importlib.reload(utils.data)

    return data.fig_kwhTon, data.fig_PrimVolts,\
         data.fig_elec, data.fig_elec1, data.fig_HeatAvgMW,\
             data.fig_Ontime_PX3, data.fig_RWI


# callback to update EAF Chemical Layout with new data
#--------------------------------------------------------------------------
@app.callback([Output('g_PPM','figure'),
            Output('g_Temp','figure'),
            Output('g_O2perTon','figure'),
            Output('g_O2scfLbC','figure')],
            [Input('EAF-chem-interval','n_intervals')])
def read_data(n):
    importlib.reload(utils.data)

    return data.fig_PPM, data.fig_Temp, data.fig_O2perTon, data.fig_O2scfLbC





# callback to update EAF Slag Layout with new data
#--------------------------------------------------------------------------
@app.callback([Output('g_B3','figure'),
            Output('g_SlagMass','figure'),
            Output('g_Lime','figure'),
            Output('g_CaO','figure'),
            Output('g_MgO','figure'),
            Output('g_FeO','figure'),
            Output('g_Al2O3','figure'),
            Output('g_SiO2','figure'),
            Output('g_Fe_Loss','figure'),
            Output('g_MgO_Loss','figure'),
            Output('g_matrix','figure'),
            Output('g_Slag_Tap_Ratio','figure'),
            Output('g_Cr_Partition','figure'),
            Output('g_Cr_Part_FeO','figure'),
            Output('g_B3_Group','figure')
            ],
            [Input('EAF-slag-interval','n_intervals')])
def read_data(n):
    importlib.reload(utils.data)

    return data.fig_B3, data.fig_SlagMass, data.fig_Lime, data.fig_CaO,\
        data.fig_MgO, data.fig_FeO, data.fig_Al2O3, data.fig_SiO2, \
        data.fig_Fe_Loss, data.fig_MgO_Loss, data.fig_matrix, \
        data.fig_Slag_Tap_Ratio, data.fig_Cr_Partition, data.fig_Cr_Part_FeO,\
        data.fig_B3_Group


# callback to update EAF Commodities Layout with new data
#--------------------------------------------------------------------------

# callback to update EAF Crew Layout with new data
#--------------------------------------------------------------------------

# callback to update EAF PX3 Layout with new data
#--------------------------------------------------------------------------

@app.callback([
            Output('g_Currents','figure'),
            Output('g_MaxTap','figure'),
            Output('g_MeanSF','figure'),
            Output('g_MWH1','figure'),
            Output('g_NCC1','figure'),
            Output('g_NCC2','figure'),
            Output('g_NCC3','figure'),
            Output('g_I2H','figure')],
            [Input('EAF-PX3-interval','n_intervals')])
def read_data(n):

    importlib.reload(utils.data)

    return data.fig_Currents,data.fig_MaxTap, data.fig_MeanSF,data.fig_MWH1,\
    data.fig_NCC1, data.fig_NCC2, data.fig_NCC3, data.fig_I2H


# I hate to have done this...but I see no other way but importing the df from frame
# and doing the figure here, instead of in data.py or in frame.py
# @app.callback(
#     [Output('g_PowerCurve','figure'),
#     Output('powercurveslider_output','children')],
#     [Input('powercurveslider', 'value'),
#     Input('dd_tap','value')])
# def update_output(Heat,tap):
 
#     importlib.reload(utils.frame)

#     dff = frame.df
#     dff['Iavg'] = (dff['Current1']+dff['Current2']+dff['Current3'])/3
#     dff = dff[dff.HeatNo >= Heat[0]].copy()
#     dff = dff[dff.HeatNo <= Heat[1]].copy()
#     dff = dff[dff.TransformerTap.isin(tap)]
#     dff['TransformerTap'] = dff['TransformerTap'].astype(str) #make color discrete
#     fig_PowerCurve = px.scatter(dff,x='Iavg',y='MWTot',color='TransformerTap',
#         color_discrete_sequence=px.colors.qualitative.T10)
#         #https://plotly.com/python/discrete-color/
    
#     string = 'Heats {} to {}'.format(Heat[0],Heat[1])

#     return fig_PowerCurve,string

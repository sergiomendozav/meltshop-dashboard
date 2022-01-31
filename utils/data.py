#%%
from doctest import DocFileTest
import pandas as pd
import numpy as np
import statistics
import math
import plotly.express as px 
import plotly.graph_objects as go


Heats = pd.read_csv('https://raw.githubusercontent.com/sergiomendozav/meltshop/master/Heats.csv')
sql_data = pd.read_csv('https://raw.githubusercontent.com/sergiomendozav/meltshop/master/sql_data.csv')
slag_samples = pd.read_csv('https://raw.githubusercontent.com/sergiomendozav/meltshop/master/slag_samples.csv')

Heats['O2scfLbC'] = Heats['TotalO2Cns']/Heats['AdditionsCarbInj']

# Dataframe merges
df = pd.merge(Heats,sql_data, on='HeatNo', how='outer')
df = pd.merge(df,slag_samples,on='HeatNo', how='outer')
df = df.sort_values(by=['HeatNo'])


# Feature Engineering (adding a bunch of columns...)
# How to look for columns with "something" in their name
# [col for col in df.columns if 'something' in col]
# or...to find a column with "xx" in its name
# df.filter(regex='Wt')

df['LimePerTon'] = df['EAF_Lime']/df['ScrapWt']
df['Al2O3_SiO2'] = df['Al2O3 (%)']+df['SiO2 (%)']
df['MeanSF'] = (df['MeanSF1']+df['MeanSF2']+df['MeanSF3'])/3
df['TapTapTime'] = df['TapTapTime'].clip(0,100) #cliping the T2T time from 0 to 100 minutes
df['TapTapTime'] = df['TapTapTime'].replace(0,np.nan)
df['Iavg'] = (df['Current1']+df['Current2']+df['Current3'])/3
df['O2perTon'] = df['TotalO2Cns']/df['ScrapWt']
df['SlagLbs_TapLbs_Ratio'] = (df['Slag_Mass_Lbs']/2000)/df['TapWeight']*100
df['Cr_Partition'] = df['Cr2O3 (%)']/df['Cr_L1']
df['Cr_Part_FeO'] = df['Cr_Partition']/df['FeO (%)']

commonGrades = ['A100BPF   ','A100B     ','85HP      ','A100      ']
commonGradesColors = ['darkred', 'orange','firebrick','forestgreen']
# Getting most common Grades and creating a "Runners" dataframe
Runners = df[df['Grade'].isin(commonGrades)].copy()

# Creating Groups of Grades as they appear in the Scrap Optimizer
HP_Group = ['30HP      ','50HP      ','85HP      ','150HP     ']
B_C_Group = ['A100B     ','A100C     ']
BPF_AMH_Group = ['A100BPF   ','AMH       ']
A200_4600V_737SH_Group = ['A200      ','4600V     ','737SH     ']
df['Group'] = df['Grade'].copy()
Mask = df['Grade'].isin(HP_Group)
df.loc[Mask,'Group'] = 'HP' 
Mask = df['Grade'].isin(B_C_Group)
df.loc[Mask,'Group'] = 'B_C'
Mask = df['Grade'].isin(BPF_AMH_Group)
df.loc[Mask,'Group'] = 'BPF_AMH'
Mask = df['Grade'].isin(A200_4600V_737SH_Group)
df.loc[Mask,'Group'] = 'A200_4600V_737SH'

B3_Group = df.groupby('Group')['B3-Ratio'].mean()
B3_Count = df.groupby('Group')['B3-Ratio'].count()
B3_list = [x.strip(' ')for x in list(B3_Group.index)]
B3_values = list(B3_Group.values)
B3_values = [round(B3,2) for B3 in B3_values]

B3_df = pd.DataFrame({'B3_values':B3_values,
                      'B3_Samples':B3_Count,
                      'B3_list':B3_list},
                      columns=['B3_values','B3_Samples','B3_list']
                      )

df['TotalHeatRWI'] = df['HeatRWI1']+df['HeatRWI2']+df['HeatRWI3']
#df.to_csv('/home/sergio/Desktop/Transfer/df.csv')

# to find a column with "xx" in its name
# df.filter(regex='Wt')

# Functions


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """
    
    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]


    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

def Control_Vars(variable):
    '''Obtains the UCLx, LCLx, UCLr, LCLr, Xbar and Rbar of a continuous variable.
    It assumes there is only one measurement per sample of the variable, n=1
    Input: variable
    Output: UCLx, LCLx, UCLr, LCLr, Xbar and Rbar
    source: https://towardsdatascience.com/quality-control-charts-guide-for-python-9bb1c859c051
    '''

    # remove the NaN from the Heats with no sample
    x = pd.Series([x for x in variable if math.isnan(x)==False])


    # Define list variable for moving ranges
    MR = [np.nan]

    # Get and append moving ranges
    i = 1
    for data in range(1, len(x)):
        MR.append(abs(x[i] - x[i-1]))
        i += 1

    # Convert list to pandas Series objects    
    MR = pd.Series(MR)

    # Concatenate mR Series with and rename columns
    data = pd.concat([x,MR], axis=1).rename(columns={0:"x", 1:"mR"})

    UCLx = statistics.mean(data['x'])+3*statistics.mean(data['mR'][1:len(data['mR'])])/1.128
    LCLx = statistics.mean(data['x'])-3*statistics.mean(data['mR'][1:len(data['mR'])])/1.128
    UCLr = statistics.mean(data['mR'][1:len(data['mR'])])+3*statistics.mean(data['mR'][1:len(data['mR'])])*0.8525
    LCLr = statistics.mean(data['mR'][1:len(data['mR'])])-3*statistics.mean(data['mR'][1:len(data['mR'])])*0.8525
    Xb = statistics.mean(data['x'])
    Rb = statistics.mean(data['mR'][1:len(data['mR'])])

    return UCLx, LCLx, UCLr, LCLr, Xb, Rb


def Control_df(source_df, var_x, var_y, *args):
    '''This function creates a separate dataframe used for the Xbar graphs
    Input: var_x, var_y
    Output: df
    '''
    df = pd.DataFrame()

    df[var_y] = source_df[var_y].copy()
    df[var_x] = source_df[var_x].copy() # commonly HeatNo
    Control_tuple = Control_Vars(source_df[var_y])
    df['UCLx'] = Control_tuple[0]
    df['LCLx'] = Control_tuple[1]
    df['UCLr'] = Control_tuple[2]
    df['LCLr'] = Control_tuple[3]
    df[var_y+'_avg'] = Control_tuple[4]
    df['r_avg'] = Control_tuple[5]
    
    for ar in args:
        df[ar] = source_df[ar].copy()
        df[ar]= df[ar].fillna(0) #replace NaN with 0.
    return df


def pxline(df,x,y,title,range_y, mode, hoverdata):
    fig = px.line(df,x=x,y=y,title=title, range_y=range_y,hover_data=hoverdata)
    fig.update_traces(mode=mode) # hovertemplate=None
    fig.update_layout(hovermode="x")
    fig.update_xaxes(tickformat="05d") # format ticket to be 5 decimals
    return fig

def Scatter(df,x,y,name,mode,xlabel,ylabel,title):
    fig = go.Figure()
    x=df[x]
    y=df[y]
    fig.add_trace(go.Scatter(x=x,y=y,name=name))
    fig.update_traces(mode=mode)
    fig.update_layout(hovermode="x",
                        xaxis_title=xlabel,
                        yaxis_title=ylabel,
                        title=title
                        )
    fig.update_xaxes(tickformat="05d") # format ticket to be 5 decimals
    return fig


def Control_graph(df,x,y,title,xaxis_title,yaxis_title,hover_var=None, color_var=None):
    '''Plots a UCL, LCL X bar type of graph of a variable
    Input: df,x,y,title,xaxis_title,yaxis_title,hover_var, color_var
    Output: figure (go.Figure())'''
    
    # this if is required in case hover_var is None, you can't .copy() a None
    # the reason for the append is because I can only send a list as optional argument
    # for the Control_df function, therefore, I have to join the hover_var with the color_var
    # and make it one list for the function argument.
    if ((hover_var != None) and (color_var != None)):
        hover_color = hover_var.copy()
        hover_color.append(color_var)

    if hover_var == None:
        if color_var == None:
            Ctrl_df = Control_df(df,x,y) 
            Ctrl_df = Ctrl_df.where(pd.notnull(Ctrl_df),None)
            fig = px.scatter(Ctrl_df,x=x,y=y)
        elif color_var != None:
            Ctrl_df = Control_df(df,x,y,color_var) 
            Ctrl_df = Ctrl_df.where(pd.notnull(Ctrl_df),None)
            fig = px.scatter(Ctrl_df,x=x,y=y,color=color_var)

    if hover_var != None:
        if color_var == None:
            Ctrl_df = Control_df(df,x,y,hover_var)
            Ctrl_df = Ctrl_df.where(pd.notnull(Ctrl_df),None)
            fig = px.scatter(Ctrl_df,x=x,y=y, hover_data=hover_var)
        elif color_var != None:
            Ctrl_df = Control_df(df,x,y,hover_color)
            Ctrl_df = Ctrl_df.where(pd.notnull(Ctrl_df),None)
            fig = px.scatter(Ctrl_df,x=x,y=y, hover_data=hover_var, color=color_var)

    fig.add_trace(
        go.Scatter(
        x=Ctrl_df[x],
        y=Ctrl_df['UCLx'],
        name = 'UCL',
        line= dict(color='red', width=1, dash='dash')))

    fig.add_trace(
        go.Scatter(
        x=Ctrl_df[x],
        y=Ctrl_df['LCLx'],
        name = 'LCL',
        line= dict(color='red', width=1, dash='dash')))

    fig.add_trace(
        go.Scatter(
        x=Ctrl_df[x],
        y=Ctrl_df[y+'_avg'],
        name = 'avg',
        line= dict(color='royalblue', width=1)))

    low_idx, high_idx = hl_envelopes_idx(Ctrl_df[y])

    fig.add_trace(
        go.Scatter(
        x=Ctrl_df[x][low_idx],
        y=Ctrl_df[y][low_idx],
        name = 'Low',
        line= dict(color='firebrick', width=1)))

    fig.add_trace(
        go.Scatter(
        x=Ctrl_df[x][high_idx],
        y=Ctrl_df[y][high_idx],
        name = 'High',
        line= dict(color='firebrick', width=1)))

    # fig.update_traces(textposition="bottom right")
    fig.update_layout(hovermode="x")
    fig.update_xaxes(tickformat="05d") # format ticket to be 5 decimals
    fig.update_layout(
    title=title,
    xaxis_title = xaxis_title,
    yaxis_title= yaxis_title)
    # https://plotly.com/python/reference/ search for coloraxis_colorbar
    fig.update_layout(coloraxis_colorbar=dict(
    title=color_var,
    thicknessmode="pixels", thickness=50,
    lenmode="pixels", len=200,
    yanchor="top", y=0.615,
    ))

    return fig


def Elec_graph(df):
    fig = go.Figure()

    if max(df['Add_A_cum']) > 0:
        A = df['Add_A_cum'].copy()
        A = list(A[A>0])
        maxa = int(max(A)) #removed the +1 since the last one doesn't mean anything
        # other than saying the last electrode add was done x number of heats.
        dictA = {}

        for i in range(1,maxa):
            dictA[i] = A.count(i)

        fig.add_trace(go.Scatter(x=list(dictA.keys()),
        y=list(dictA.values()),name='A'))

    if max(df['Add_B_cum']) > 0:
        B = df['Add_B_cum'].copy()
        B = list(B[B>0])
        maxb = int(max(B))
        dictB = {}

        for i in range(1,maxb):
            dictB[i] = B.count(i)

        fig.add_trace(go.Scatter(x=list(dictB.keys()),
        y=list(dictB.values()),name='B'))


    if max(df['Add_C_cum']) > 0:
        C = df['Add_C_cum'].copy()
        C = list(C[C>0])
        maxc = int(max(C))
        dictC = {}

        for i in range(1,maxc):
            dictC[i] = C.count(i)

        fig.add_trace(go.Scatter(x=list(dictC.keys()),
        y=list(dictC.values()),name='C'))

    
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    #fig.update_layout(hovermode="x")
    fig.update_layout(
    title='Electrodes Heats/Add ',
    xaxis_title = 'Electrode Add Number',
    yaxis_title= 'Heats/Add',
    legend_title= 'Electrode'
    )

    return fig

def Elec_graph_1(df):
    fig_elec = go.Figure()
    fig_elec.add_trace(go.Scatter(x=df['HeatNo'],
    y=df['Add_A_cum'],
    name='A'))
    fig_elec.add_trace(go.Scatter(x=df['HeatNo'],
    y=df['Add_B_cum'],
    name='B'))
    fig_elec.add_trace(go.Scatter(x=df['HeatNo'],
    y=df['Add_C_cum'],
    name='C'))
    fig_elec.update_traces(mode="markers+lines", hovertemplate=None)
    fig_elec.update_layout(hovermode="x")
    fig_elec.update_xaxes(tickformat="05d") # format ticket to be 5 decimals
    fig_elec.update_layout(
    title='Electrode Cumulative Additions',
    xaxis_title = 'HeatNo',
    yaxis_title= 'No. of Electrodes',
    legend_title= 'Phase')

    return fig_elec

#%%
# Figures

# Electrical Figures
#--------------------------------------------------------------------------
fig_kwhTon = pxline(df, x='HeatNo', y='OperKwhPerTon', title='kWh/Ton', range_y=[300,550], mode='lines', hoverdata=None)
fig_PrimVolts = pxline(df, x='HeatNo', y='PrimaryVolts', title='Primary Volts', range_y = [13.0,14.5], mode='lines', hoverdata=None)
fig_elec = Elec_graph(df)
fig_elec1 = Elec_graph_1(df)
fig_HeatAvgMW = pxline(df, x='HeatNo', y='HeatAvgMW', title='Heat MW', range_y=[20,40], mode='lines', hoverdata=None)
fig_Ontime_PX3 = Scatter(df,'HeatNo', 'Ontime_PX3','PON Time','lines','HeatNo','Minutes','Heat Times')
fig_Ontime_PX3.add_trace(
    go.Scatter(x=df['HeatNo'],y=df['TapTapTime'],name='T2T Time'))
fig_Ontime_PX3.add_trace(
    go.Scatter(x=df['HeatNo'],y=df.Ontime_PX3.rolling(window=14).std(),
    name='PON StDev'))
fig_RWI = Control_graph(df,'HeatNo','TotalHeatRWI','Heat RWI','HeatNo','Heat RWI')

#1st HEG Electrode Phase C on Heat 30450
# fig_elec1.add_vline(x=30450,line_width=2, 
# line_dash='dash',line_color='green',
#  annotation_text="1st HEG Phace C @ 30450", 
# annotation_position="bottom left",
# annotation_font_size=12,
# annotation_font_color="green")


# # adding when the SVC was out
# if df['HeatNo'].min() < 30234:
#     fig_PrimVolts.add_vrect(x0=30234,x1=30243,row='all',col=1,annotation_text = 'No SVC',
#     annotation_position='top left',fillcolor='green',opacity=0.25,line_width=0)
# elif (df['HeatNo'].min() >= 30234) and (df['HeatNo'].min()<30243):
#     x0 = df['HeatNo'].min()
#     fig_PrimVolts.add_vrect(x0=x0,x1=30243,row='all',col=1,annotation_text = 'No SVC',
#     annotation_position='top left',fillcolor='green',opacity=0.25,line_width=0)


                        

# Chemical
#--------------------------------------------------------------------------
fig_PPM = Control_graph(df,'HeatNo','Oxy','First PPM Reading','HeatNo','O2 PPM',color_var='OTemp')
fig_PPM.add_vline(x=31165,line_width=2, 
line_dash='dash',line_color='green',
 annotation_text="O2 Trial ->", 
annotation_position="bottom left",
annotation_font_size=12,
annotation_font_color="green")
fig_PPM.update_coloraxes(dict(cmin=2800, cmax=3100))

fig_Temp = Control_graph(df,'HeatNo','OTemp','First Temp Reading','HeatNo','Temperature')
fig_Temp.add_vline(x=31165,line_width=2, 
line_dash='dash',line_color='green',
 annotation_text="O2 Trial ->", 
annotation_position="bottom left",
annotation_font_size=12,
annotation_font_color="green")


fig_O2perTon = Control_graph(df,'HeatNo','O2perTon','O2 scf/ton','HeatNo','SCF/Ton')
fig_O2scfLbC = px.histogram(df,x='O2scfLbC',color='Grade', 
                            title='O2 scf / Lb Inj. Carbon',
                            labels={'O2scfLbC':'O2 scf / Lb C'})


# Slag Figures
#--------------------------------------------------------------------------
fig_Lime = Control_graph(df,'HeatNo','LimePerTon','Lime [Lbs/cTon]','HeatNo','Lbs/cTon',hover_var=['CaO (%)'], color_var='B3-Ratio')
fig_B3 = Control_graph(df,'HeatNo','B3-Ratio','B3-Ratio','HeatNo','B3-Ratio')
fig_SlagMass = Control_graph(df,'HeatNo','Slag_Mass_Lbs','Slag Mass Lbs','HeatNo','Lbs.',hover_var=['B3-Ratio'])
fig_CaO = Control_graph(df,'HeatNo','CaO (%)','CaO %','HeatNo','%',hover_var=['B3-Ratio'])
fig_MgO = Control_graph(df,'HeatNo','MgO (%)','MgO %','HeatNo','%',hover_var=['B3-Ratio'])
fig_FeO = Control_graph(df,'HeatNo','FeO (%)','FeO %','HeatNo','%',hover_var=['B3-Ratio'])
fig_Al2O3 = Control_graph(df,'HeatNo','Al2O3 (%)','Al2O3 %','HeatNo','%',hover_var=['B3-Ratio'])
fig_SiO2 = Control_graph(df,'HeatNo','SiO2 (%)','SiO2 %','HeatNo','%',hover_var=['B3-Ratio','CaO (%)'])
fig_Fe_Loss = Control_graph(df,'HeatNo','Fe_Lbs_Loss_in_Slag','Fe [Lbs] loss in Slag','HeatNo','Lbs.',hover_var=['B3-Ratio'])
fig_MgO_Loss = Control_graph(df,'HeatNo','MgO_Lbs_Refr_Loss','MgO [Lbs] Refractory Loss','HeatNo','Lbs.',hover_var=['B3-Ratio'])
fig_matrix = px.scatter_matrix(Runners,dimensions=['FeO (%)','Al2O3 (%)','CaO (%)','B3-Ratio'],color='Grade',title='Slag Matrix (Main Grades)',hover_data=['SiO2 (%)'])
fig_Slag_Tap_Ratio  = Control_graph(df,'HeatNo','SlagLbs_TapLbs_Ratio','SlagMass/Tap Ratio','HeatNo','SlagMass/Tap Ratio')
fig_Slag_Tap_Ratio.add_hrect(y0=9.5, y1=10.0, line_width=0, fillcolor="green", opacity=0.2)
fig_Cr_Partition = Control_graph(df,'HeatNo','Cr_Partition','Cr Partition','HeatNo','Cr[slag]/Cr[L1]',hover_var=['Cr2O3 (%)','Cr_L1','Grade'])
fig_Cr_Part_FeO = Control_graph(df,'HeatNo','Cr_Part_FeO',
'Cr Partition / FeO (%)','HeatNo','Cr Partition/FeO %',hover_var=['Cr2O3 (%)','Cr_L1','FeO (%)','Grade'])
fig_B3_Group = px.bar(B3_df,x = 'B3_list',y = 'B3_values',
text = 'B3_values', color='B3_Samples')
fig_B3_Group.update_traces(texttemplate='%{text}', textposition='outside')
fig_B3_Group.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',title='B3 Groups',yaxis_range=[0,3])



# updating fig_CaO,fig_SiO2,fig_Al2O3 to include average of the particular figure by grade
figures = [fig_CaO,fig_SiO2,fig_Al2O3]
element = ['CaO (%)', 'SiO2 (%)', 'Al2O3 (%)']
for fig, elem in zip(figures,element):
    for grade, color in zip(commonGrades,commonGradesColors):
        x = Runners['HeatNo'].copy()
        # get the mean of the "elem" element for the "grade" grade
        y= Runners.loc[Runners['Grade']==grade,elem].mean()
        fig.add_trace(
                go.Scatter(
                    x=x,
                    # this comprehensive list is to avoid the graph plotting just one point
                    # instead if draws a line for the lenght of "x" (HeatNo).
                    y=[y for i in range(len(x))], 
                    name = grade.strip(), #remove the leading spaces
            line= dict(color=color, width=1)))


# # 30425 started Lime in Lbs/CTon
# fig_Lime.add_vline(x=30425,line_width=2, 
# line_dash='dash',line_color='green',
#  annotation_text="Lime in Lbs/cTon ->", 
# annotation_position="bottom left",
# annotation_font_size=12,
# annotation_font_color="green")

# fig_B3.add_vline(x=30425,line_width=2, 
# line_dash='dash',line_color='green',
#  annotation_text="Lime in Lbs/cTon ->", 
# annotation_position="bottom left",
# annotation_font_size=12,
# annotation_font_color="green")




# df['CaO + MgO (%)'] = df['CaO (%)']+df['MgO (%)']
# df['SiO2 + Al2O3'] = df['SiO2 (%)']+df['Al2O3 (%)']

# fig = px.scatter_ternary(df,a='SiO2 + Al2O3',b='CaO + MgO (%)',c='FeO (%)')
# fig.show()

# fig = px.scatter_ternary(df,a='SiO2 (%)',b='CaO (%)',c='FeO (%)')
# fig.show()

# fig = px.scatter_ternary(df,a='FeO (%)',b='CaO (%)',c='Al2O3 (%)')
# fig.show()

# fig = px.scatter_ternary(df,a='FeO (%)',b='CaO (%)',c='SiO2 (%)')
# fig.show()

# fig = px.scatter_ternary(df,a='FeO (%)',b='CaO (%)',c='MgO (%)')
# fig.show()


# fig = px.scatter_ternary(df,a='FeO (%)',b='SiO2 (%)',c='CaO (%)')
# fig.show()

# Commodities
#--------------------------------------------------------------------------


# Crew
#--------------------------------------------------------------------------

# PX3
#--------------------------------------------------------------------------
fig_Currents = Scatter(df,'HeatNo','Current1','Current 1','lines','HeatNo','kA','Currents')
fig_Currents.add_trace(go.Scatter(x = df['HeatNo'],y = df['Current2'],name = 'Current 2'))
fig_Currents.add_trace(go.Scatter(x = df['HeatNo'],y = df['Current3'],name = 'Current 3'))
fig_MaxTap = Scatter(df,'HeatNo','TransformerTap','Max Transformer Tap','markers+lines','HeatNo','Max Tap','Max Transformer Tap')
fig_MeanSF = Control_graph(df,'HeatNo','MeanSF','Mean Stability Factor','HeatNo','Mean SF')
fig_MWH1 = Control_graph(df,'HeatNo','MWH1','MWH 1st Bucket','HeatNo','MWH 1st Bucket')
fig_NCC1 = pxline(df, x='HeatNo', y='NCCcounter1', title='NCC Phase A', range_y = [0,200], mode='markers+lines', hoverdata=None)
fig_NCC2 = pxline(df, x='HeatNo', y='NCCcounter2', title='NCC Phase B', range_y = [0,200], mode='markers+lines', hoverdata=None)
fig_NCC3 = pxline(df, x='HeatNo', y='NCCcounter3', title='NCC Phase C', range_y = [0,200], mode='markers+lines', hoverdata=None)
fig_I2H = Scatter(df,'HeatNo','I2H1','I2H1','markers','HeatNo','I2H','I2H')
fig_I2H.add_trace(go.Scatter(x=df['HeatNo'],y=df['I2H2'],name='I2H2',mode='markers'))
fig_I2H.add_trace(go.Scatter(x=df['HeatNo'],y=df['I2H3'],name='I2H3',mode='markers'))


                # fig = px.scatter(x=df['OperKwhPerTon'],y=df['HeatAvgMW'], 
                # marginal_x='histogram',
                # marginal_y='histogram')
                # fig.show()








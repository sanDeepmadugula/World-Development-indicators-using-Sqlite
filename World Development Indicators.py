#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sqlite3
import os
from bokeh.plotting import figure, show
#from bokeh.charts import Bar
from bokeh.io import output_notebook
output_notebook()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')


# In[3]:


conn = sqlite3.connect('C:\\Analytics\\MachineLearning\\world development indicators\\database.sqlite')


# In[4]:


c = conn.cursor()


# In[5]:


for row in c.execute(
                          """
                          Select *
                          from Country
                          Limit 3
                          """):
    print(row)


# In[6]:


# Store Country data in a pandas dataframe via a SQL query
Country = pd.read_sql(
           """
           Select * 
           from Country
           """, con=conn)


# In[7]:


Country.head(2)


# In[9]:


pd.read_sql(
         """
         select * 
         from Country
         limit 3
         """,con=conn)


# In[10]:


Country[Country['CountryCode']== 'AFG']


# In[11]:


Sounth_Asia = Country[Country['Region']== 'South Asia']
Sounth_Asia


# In[12]:


# where clause
pd.read_sql(

           """
           select * 
           from Country
           where CountryCode = 'AFG'
           """, con=conn)


# In[13]:


# SQL WHERE clause with additional conditionals

pd.read_sql(

            '''
            Select ShortName, LongName, IncomeGroup, LatestPopulationCensus
            From Country
            where LatestPopulationCensus > 1999 
            AND(Region = 'South Asia' OR Region = 'Europe & Central Asia')
            AND(NOT ShortName = 'Albania') AND(NOT IncomeGroup='High income:OECD')
            ''', con=conn)


# In[14]:


# Group by
pd.read_sql(

            """
            Select Region,
            Count(*) as [Count]
            from Country
            group by Region
            order by 2 desc
            """, con=conn)


# In[15]:


pd.read_sql(

      """
      select IncomeGroup,
      Count(*) as [Count]
      from Country
      group by IncomeGroup
      order by 1 asc
      """, con=conn)


# In[19]:


pd.read_sql(
        """ 
           
            SELECT      A.CountryCode
                        ,B.LatestPopulationCensus
                        ,B.SourceOfMostRecentIncomeAndExpenditureData
                        ,B.ShortName
            FROM       ( 
                            -- First subquery (i.e the Left table)
                            
                           SELECT      CountryCode
                                        ,LatestPopulationCensus
                                        ,SourceOfMostRecentIncomeAndExpenditureData
                                        ,ShortName
                           FROM        Country
                           WHERE       CountryCode IN ('AFG','ALB', 'ASM', 'BEL')
                        ) AS A
         LEFT JOIN   (
                            -- Second subquery (i.e the right table )
                            
                            SELECT      CountryCode
                                        ,LatestPopulationCensus
                                        ,SourceOfMostRecentIncomeAndExpenditureData
                                        ,ShortName
                            FROM        Country AS A
                            WHERE       CountryCode IN ('AFG','ARM', 'URY', 'BEL')
                            
                          ) AS B
            ON          A.CountryCode = B.CountryCode    
            
        """, con=conn)


# In[20]:


pd.read_sql(

    """
    Select A.ShortName, A.LongName, B.LatestPopulationCensus
    
    from (
     (select * from Country where Region = 'South Asia' and not ShortName='Nepal')
     
     As a 
     
     Left join(
      select * from Country where Region = 'South Asia' and not ShortName='Afghanistan'
     ) 
     AS B
     on A.CountryCode = B.CountryCode)
    """, con=conn)


# In[21]:


# union
pd.read_sql(

      """
      Select CountryCode,
      LatestPopulationCensus,
      SourceOfMostRecentIncomeAndExpenditureData,
      ShortName
      from Country
      where CountryCode in ('AFG', 'ALB', 'ASM','BEL')
      
      UNION
      
      Select CountryCode,
      LatestPopulationCensus,
      SourceOfMostRecentIncomeAndExpenditureData,
      ShortName
      from Country
      where CountryCode in ('AFG', 'ALB', 'ASM','BEL')
      
      """,con=conn)


# In[22]:


pd.read_sql(

     '''
     Select CountryCode, ShortName, LatestPopulationCensus
     
     from Country
     where Region = 'South Asia'
     
     Union
     
     Select CountryCode, LatestPopulationCensus, LongName
     
     from Country
     
     where Region = 'North America'
     ''',con=conn)


# In[23]:


# Intersect
pd.read_sql(
        """ 
                           SELECT      CountryCode
                                        ,LatestPopulationCensus
                                        ,SourceOfMostRecentIncomeAndExpenditureData
                                        ,ShortName
                           FROM        Country
                           WHERE       CountryCode IN ('AFG','ALB', 'ASM', 'BEL')
                       
                           INTERSECT
                           
                           SELECT      CountryCode
                                        ,LatestPopulationCensus
                                        ,SourceOfMostRecentIncomeAndExpenditureData
                                        ,ShortName
                           FROM        Country AS A
                           WHERE       CountryCode IN ('AFG','ARM', 'URY', 'BEL')
            
        """, con=conn)


# # Data Analysis and Visualization

# In[25]:


Indicators = pd.read_sql(""" SELECT   * 
                             FROM     Indicators 
                             WHERE    IndicatorCode IN 
                                      (  'AG.LND.PRCP.MM, AG.LND.FRST.K2'
                                       , 'EG.ELC.ACCS.ZS', 'EG.ELC.FOSL.ZS'
                                       , 'EN.POP.DNST', 'SG.VAW.REAS.ZS'
                                       , 'SM.POP.NETM', 'SP.POP.65UP.TO.ZS'
                                       , 'FI.RES.TOTL.DT.ZS', 'GC.DOD.TOTL.GD.ZS'
                                       , 'MS.MIL.XPND.GD.ZS','SI.POV.GINI'
                                       , 'IP.JRN.ARTC.SC', 'SE.ADT.1524.LT.ZS'
                                      )  
                        """, con=conn)
Indicators


# In[26]:


#List of indicators and number of entries for each
AllIndicators = pd.read_sql(

              """
              select IndicatorName, IndicatorCode,
              Count(*) as [No.of entries]
              from Indicators
              group by IndicatorName
              
              """, con=conn)

AllIndicators


# In[27]:


# GINI index analysis
gini = Indicators[Indicators['IndicatorCode'] == 'SI.POV.GINI']
gini.head(5)


# In[28]:


gini[gini['CountryCode']=='AUS'].Year.unique()


# In[29]:


gini.CountryCode.unique()


# In[31]:


#Plotting a subplot of the seaborn regplot
f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3,figsize=(12,10))

points = ax1.scatter(gini[gini['CountryCode']=='CHN']['Year'], 
                    gini[gini['CountryCode']=='CHN']['Value'],
                    c=gini[gini['CountryCode']=='CHN']['Value'],s=100,cmap='viridis')
sns.regplot('Year', 'Value', data=gini[gini['CountryCode']=='CHN'],ax=ax1)
ax1.set_title('GINI Index of China')

# Plot of GINI of Argentina
points = ax2.scatter(gini[gini['CountryCode'] == 'ARG']["Year"], gini[gini['CountryCode'] == 'ARG']["Value"],
                     c=gini[gini['CountryCode'] == 'ARG']["Value"], s=85, cmap="viridis")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'ARG'], ax=ax2)
ax2.set_title("GINI Index of Argentina")

points = ax3.scatter(gini[gini['CountryCode'] == 'UGA']["Year"], gini[gini['CountryCode'] == 'UGA']["Value"],
                     c=gini[gini['CountryCode'] == 'UGA']["Value"], s=100, cmap="afmhot")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'UGA'], ax=ax3)
ax3.set_title("GINI Index of Uganda")

points = ax4.scatter(gini[gini['CountryCode'] == 'USA']["Year"], gini[gini['CountryCode'] == 'USA']["Value"],
                     c=gini[gini['CountryCode'] == 'USA']["Value"], s=100, cmap="Purples_r")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'USA'], ax=ax4)
ax4.set_title("GINI Index of USA")

points = ax5.scatter(gini[gini['CountryCode'] == 'COL']["Year"], gini[gini['CountryCode'] == 'COL']["Value"],
                     c=gini[gini['CountryCode'] == 'COL']["Value"], s=100, cmap="YlOrBr")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'COL'], ax=ax5)
ax5.set_title("GINI Index of Colombia")

points = ax6.scatter(gini[gini['CountryCode'] == 'RWA']["Year"], gini[gini['CountryCode'] == 'RWA']["Value"],
                     c=gini[gini['CountryCode'] == 'RWA']["Value"], s=100, cmap="Blues")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'RWA'], ax=ax6)
ax6.set_title("GINI Index of Rwanda")

points = ax7.scatter(gini[gini['CountryCode'] == 'RUS']["Year"], gini[gini['CountryCode'] == 'RUS']["Value"],
                     c=gini[gini['CountryCode'] == 'RUS']["Value"], s=100, cmap="Blues")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'RUS'], ax=ax7)
ax7.set_title("GINI Index of Russia")

points = ax8.scatter(gini[gini['CountryCode'] == 'ECU']["Year"], gini[gini['CountryCode'] == 'ECU']["Value"],
                     c=gini[gini['CountryCode'] == 'ECU']["Value"], s=100, cmap="winter")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'ECU'], ax=ax8)
ax8.set_title("GINI Index of Ecuador")

points = ax9.scatter(gini[gini['CountryCode'] == 'CAF']["Year"], gini[gini['CountryCode'] == 'CAF']["Value"],
                     c=gini[gini['CountryCode'] == 'CAF']["Value"], s=100, cmap="magma")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'CAF'], ax=ax9)
ax9.set_title("GINI Index of Central African Republic")
sns.set_style(style="dark")
plt.tight_layout()


# In[32]:


# take away the plots

pd.read_sql("""

          select IndicatorName, IndicatorCode, Count(*) 
          as [No.of entries]
          from Indicators
          where IndicatorName LIKE '%poverty%' or
          IndicatorName LIKE '%gini%' or
          IndicatorCode LIKE '%POV%'
          Group by IndicatorName
 
        """, con=conn)


# In[33]:


povGap = pd.read_sql('''
            
            select * 
            from Indicators
            where IndicatorCode = 'SI.POV.DDAY'
      ''', con=conn)

indexOfNames = pd.read_sql('''
           
            select CountryName, CountryCode
            from Indicators
            where IndicatorCode = 'SI.POV.DDAY'
            group by CountryCode
          ''', con=conn)
indexOfNames.tail(10)


# In[34]:


f, ((ax1, ax2, ax3), (ax4,ax5,ax6),(ax7, ax8, ax9)) = plt.subplots(3,3, figsize=(15,10))


#rus chn pol
# Plot of GINI index of China
points = ax1.scatter(povGap[povGap['CountryCode'] == 'CHN']["Year"], povGap[povGap['CountryCode'] == 'CHN']["Value"],
                     c=povGap[povGap['CountryCode'] == 'CHN']["Value"], s=100, cmap="viridis")
sns.regplot("Year", "Value", data=povGap[povGap['CountryCode'] == 'CHN'], ax=ax1)
ax1.set_title("Pov.Gap Index of China")

points = ax2.scatter(povGap[povGap['CountryCode'] == 'RUS']["Year"], povGap[povGap['CountryCode'] == 'RUS']['Value'],
                     c=povGap[povGap['CountryCode'] == 'RUS']['Value'], s= 100, cmap="magma")
sns.regplot("Year", "Value", data=povGap[povGap['CountryCode'] == 'RUS'], ax=ax2)
ax2.set_title("Pov. Gap Index Rusia")

CodeName = 'URY'
points = ax3.scatter(povGap[povGap['CountryCode'] == CodeName ]['Year'], povGap[povGap['CountryCode'] == CodeName ]['Value'],
                    c=povGap[povGap['CountryCode'] == CodeName]['Value'], s=100, cmap="viridis")
sns.regplot("Year", "Value", data=povGap[povGap['CountryCode'] == CodeName], ax=ax3)
ax3.set_title("Pov. Gap Index Uruguay")

CodeName = 'ARG'
points = ax4.scatter(povGap[povGap['CountryCode'] == CodeName ]['Year'], povGap[povGap['CountryCode'] == CodeName ]['Value'],
                    c=povGap[povGap['CountryCode'] == CodeName]['Value'], s=100, cmap="viridis")
#sns.regplot("Year", "Value", data=povGap[povGap['CountryCode'] == CodeName], ax=a4)
ax4.set_title("Pov. Gap Index Argentia")

CodeName = 'BRA'
points = ax5.scatter(povGap[povGap['CountryCode'] == CodeName ]['Year'], povGap[povGap['CountryCode'] == CodeName ]['Value'],
                    c=povGap[povGap['CountryCode'] == CodeName]['Value'], s=100, cmap="viridis")
sns.regplot("Year", "Value", data=povGap[povGap['CountryCode'] == CodeName], ax=ax5)
ax5.set_title("Pov. Gap Index Brazil")

CodeName = 'COL'
points = ax6.scatter(povGap[povGap['CountryCode'] == CodeName ]['Year'], povGap[povGap['CountryCode'] == CodeName ]['Value'],
                    c=povGap[povGap['CountryCode'] == CodeName]['Value'], s=100, cmap="viridis")
#sns.regplot("Year", "Value", data=povGap[povGap['CountryCode'] == CodeName], ax=ax6)
ax6.set_title("Pov. Gap Index Columbia")

CodeName = 'IND'
points = ax7.scatter(povGap[povGap['CountryCode'] == CodeName ]['Year'], povGap[povGap['CountryCode'] == CodeName ]['Value'],
                    c=povGap[povGap['CountryCode'] == CodeName]['Value'], s=100, cmap="viridis")
#sns.regplot("Year", "Value", data=povGap[povGap['CountryCode'] == CodeName], ax=ax7)
ax7.set_title("Pov. Gap Index India")

CodeName = 'UKR'
points = ax8.scatter(povGap[povGap['CountryCode'] == CodeName ]['Year'], povGap[povGap['CountryCode'] == CodeName ]['Value'],
                    c=povGap[povGap['CountryCode'] == CodeName]['Value'], s=100, cmap="viridis")
#sns.regplot("Year", "Value", data=povGap[povGap['CountryCode'] == CodeName], ax=ax8)
ax8.set_title("Pov. Gap Index Ukraine")

CodeName = 'TUR'
points = ax9.scatter(povGap[povGap['CountryCode'] == CodeName ]['Year'], povGap[povGap['CountryCode'] == CodeName ]['Value'],
                    c=povGap[povGap['CountryCode'] == CodeName]['Value'], s=100, cmap="viridis")
sns.regplot("Year", "Value", data=povGap[povGap['CountryCode'] == CodeName], ax=ax9)
ax9.set_title("Pov. Gap Index Turkey")






sns.set_style(style="dark")
plt.tight_layout()


# In[35]:


data = Indicators[Indicators['IndicatorCode'] == 'SE.ADT.1524.LT.ZS'][Indicators['Year']==1990]
x,y = (list(x) for x in zip(*sorted(zip(data['Value'].values, data['CountryName'].values),
                                   reverse=False)))

trace2 = go.Bar(

   x=x,
   y = y,
   marker = dict(
    color=x,
    colorscale='Portland',
    reversescale=True),
    name='Percentage of youth Literacy Rate',
    orientation='h',)
layout = dict(
    title='Barplot of Youth Literacy Rate in 1990',
     width = 680, height = 1500,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
#         domain=[0, 0.85],
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')

# Barplot of Youth literacy rates in 2010
data = Indicators[Indicators['IndicatorCode'] == 'SE.ADT.1524.LT.ZS'][Indicators['Year'] == 2010]
x, y = (list(x) for x in zip(*sorted(zip(data['Value'].values, data['CountryName'].values), 
                                                            reverse = False)))

# Plotting using Plotly 
trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Portland',
        reversescale = True
    ),
    name='Percentage of Youth Literacy Rate',
    orientation='h',
)

layout = dict(
    title='Barplot of Youth Literacy Rate in 2010',
     width = 680, height = 1500,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
#         domain=[0, 0.85],
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')


# In[36]:


# women believe husband is justified in beating the wife
data = Indicators[Indicators['IndicatorCode']== 'SG.VAW.REAS.ZS']


# In[37]:


data = Indicators[Indicators['IndicatorCode']=='SE.ADT.1524.LT.ZS']
data['Year'] = [str(x) for x in data['Year']]
years = ['2000',
 '2001',
 '2002',
 '2003',
 '2004',
 '2005',
 '2006',
 '2007',
 '2008',
 '2009',
 '2010',
 '2011',
 '2012',
 '2013',
 '2014']

country = ['Burkina Faso', 'Central African Republic', 'Kuwait', 'Turkey',
       'United Arab Emirates', 'Uruguay', 'Bolivia', 'Cameroon',
       'Egypt, Arab Rep.', 'Iran, Islamic Rep.', 'Mali', 'New Caledonia',
       'Swaziland', 'Tonga', 'Maldives', 'Poland', 'Rwanda', 'Afghanistan',
       'Benin', 'Burundi', 'Guinea-Bissau', 'Jordan', 'Vanuatu', 'Vietnam',
       'American Samoa', 'Argentina', 'Brazil', 'Comoros', 'Guam',
       'Hungary', 'Indonesia', 'Malaysia', 'Mexico', 'Mozambique', 'Palau',
       'Panama', 'Philippines', 'Puerto Rico', 'Singapore', 'South Africa',
       'Thailand', 'Trinidad and Tobago', 'Bahrain', 'Bangladesh',
       'Brunei Darussalam', 'Cuba', 'Dominican Republic', 'Greece',
       'India', 'Italy', 'Macao SAR, China', 'Nepal', 'Pakistan', 'Peru',
       'Portugal', 'Sao Tome and Principe', 'Spain', 'Sri Lanka',
       'Syrian Arab Republic', 'Venezuela, RB', 'Chile', 'China',
       'Ecuador', 'Haiti', 'Morocco', 'Paraguay', 'Zimbabwe', 'Israel',
       'Myanmar', 'Costa Rica', 'Liberia', 'Libya', 'Tunisia', 'Malta',
       'Qatar', 'Algeria', 'Malawi', 'Seychelles', "Cote d'Ivoire",
       'Senegal', 'Tanzania', 'Armenia', 'Belarus', 'Estonia',
       'Kazakhstan', 'Latvia', 'Lithuania', 'Moldova','Lesotho', 'Madagascar', 'Mauritania', 'Mongolia',
       'Papua New Guinea', 'Sudan', 'Togo', 'Uzbekistan', 'Albania',
       'Angola', 'Bulgaria', 'Congo, Dem. Rep.', 'Honduras', 'Nicaragua',
       'Niger', 'Ukraine', 'Eritrea', 'Georgia', 'Oman', 'Sierra Leone',
       'Suriname', 'Bhutan', 'Cayman Islands', 'Lebanon',
       'Korea, Dem. Rep.', 'South Sudan', 'Guyana', 'Timor-Leste',
       'Congo, Rep.', 'Montenegro', 'Serbia', 'Austria']


# In[39]:


from math import pi

from bokeh.io import show
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
# data = data.set_index('Year')
# this is the colormap from the original NYTimes plot
colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors)
# Set up the data for plotting. We will need to have values for every
# pair of year/month names. Map the rate to a color.
countr = []
year = []
color = []
rate = []
for y in years:
    for m in country:
        countr.append(m)
        year.append(y)
#         d[(d['x']>2) & (d['y']>7)]
        monthly_rate = data[(data['CountryName']==m) & (data['Year']==y)]['Value']
        rate.append(monthly_rate)

source = ColumnDataSource(
    data=dict(country=countr, year=year, rate=rate)
)

TOOLS = "hover,save,pan,box_zoom,wheel_zoom"

p = figure(title="Women who believe Husbands are justified in beating wifes",
           x_range=years, y_range=list(reversed(country)),
           x_axis_location="above", plot_width=900, plot_height=900,
           tools=TOOLS)

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = pi / 3

p.rect(x="year", y="country", width=1, height=1,
       source=source,
       fill_color={'field': 'rate', 'transform': mapper},
       line_color=None)

p.select_one(HoverTool).tooltips = [
#     ('date', '@countr @year'),
    ('rate', '@rate'),
]

show(p)      # show the plot


# In[ ]:





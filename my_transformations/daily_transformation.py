#!/usr/bin/env python3
# daily_transformation.py
#
# Author: Kevin Graves
# Date Modified: 5/4/2020
# Description: Script to scrape newest COVID-19 data from
#               github.com/CSSEGISandData/COVID-19/, save it locally,
#               and process it to allow for more rapid plotting
#               
# Notes: Eventually should change this to run this as an iterative update
#################### Imports ##################################################
 
import numpy as np
import time
import sys
import csv
import requests
import pandas as pd
import os
import datetime
import json
from scipy.optimize import curve_fit


#################### Parameters ###############################################
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50) 

start_date = datetime.date(2020,1,22)
end_date   = datetime.date.today() - datetime.timedelta(days=1)
base_url = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/"+
            "master/csse_covid_19_data/csse_covid_19_daily_reports/{0}.csv")
# Where to save the data from github so you don't have to always grab the data from Github
data_dir = "./../csse_covid_19_data/csse_covid_19_daily_reports/"

# list of the different ways they upload the data to github over time
# (only add to the END of this list if more file type are added)
f_types = [["Province/State","Country/Region","Last Update","Confirmed","Deaths","Recovered"],
           ["Province/State","Country/Region","Last Update","Confirmed",
            "Deaths","Recovered","Latitude","Longitude"],
           ["FIPS","Admin2","Province_State","Country_Region","Last_Update",
            "Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key"],
           ['FIPS','Admin2','Province_State','Country_Region','Last_Update','Lat',
            'Long_','Confirmed','Deaths','Recovered','Active','Combined_Key','Incidence_Rate','Case-Fatality_Ratio']]
# Try to not change this too much - adding columns is fine, but don't change the
#  names, it will break stuff (e.g. when plotting, and joining different files together)
df_columns = ["FIPS","County","Province/State","Country/Region","Updated Date/Time","Date",
            "Latitude","Longitude","Confirmed","Deaths","Recovered","Active","Combined_Key"]
# They had massive ADHD when it comes to date formats, here are the ones they have used (so far)
date_formats = ['%m/%d/%Y %H:%M','%m/%d/%y %H:%M','%Y-%m-%dT%H:%M:%S','%Y-%m-%d %H:%M:%S']
# Pickle file names for foundational dataframes
base_county_pickle_name = "county_df_{0}.pkl"
base_state_pickle_name = "state_df_{0}.pkl"
# Number of Days in slider
n_day_slider = 7
#################### Functions ################################################
def init():
    """
    Create the data directory is it does not exist
    """

    try:
        os.mkdir(data_dir)
    except FileExistsError:
        pass

def get_data():
    """
    Get the foundational dataframes either from local files/pickles or from
    the source on Github

    - Grab data from this forked repo
    - Build and return a single foundational dataframe
    """


    loop_date = start_date

    df = pd.DataFrame(columns=df_columns)
    n_loop = 0
    while loop_date <= end_date:
        date_str = loop_date.strftime("%m-%d-%Y")
        print(date_str)
        l_local = True
        try:
            # Try to open file from local data directory
            fin = open(data_dir+date_str+".csv",'r',encoding='utf-8-sig')
            lines = fin.readlines()
        except FileNotFoundError:
            # Otherwise grab from github
            print("Error: No file {0} found in {1}".format(date_str+".csv",data_dir))
            sys.exit(1)
        # Grab header
        header_vals = lines[0].strip().split(',')
       
        # Assert that this header is in list (this will throw an error if the header is not
        #  in f_types (e.g. they changes the names of the columns)
        ind = f_types.index(header_vals)
       
           
        # remove header row to make indices line up nicely
        del lines[0]
        
        # Some stuff to make the different file formats exist in one df
        if ind == 0:
            str_inds = [0,1]
            date_ind = 2
            int_inds = [3,4,5]
            float_inds = []
            parsed_lines = [parse_row(r.strip(),str_inds,date_ind,int_inds,float_inds) for r in lines]
            df_append = pd.DataFrame(parsed_lines,
                                     columns=("Province/State","Country/Region",
                                              "Updated Date/Time","Confirmed","Deaths",
                                              "Recovered"))
        elif ind == 1:
            str_inds = [0,1]
            date_ind = 2
            int_inds = [3,4,5]
            float_inds = [6,7]
            parsed_lines = [parse_row(r.strip(),str_inds,date_ind,int_inds,float_inds) for r in lines]
            df_append = pd.DataFrame(parsed_lines,
                                     columns=("Province/State","Country/Region",
                                              "Updated Date/Time","Confirmed","Deaths",
                                              "Recovered","Latitude","Longitude"))
        elif ind == 2:
            str_inds = [0,1,2,3,11]
            date_ind = 4
            int_inds = [7,8,9,10]
            float_inds = [5,6]
            parsed_lines = [parse_row(r.strip(),str_inds,date_ind,int_inds,float_inds) for r in lines]
            df_append = pd.DataFrame(parsed_lines,
                                     columns=("FIPS","County","Province/State",
                                              "Country/Region","Updated Date/Time",
                                              "Latitude","Longitude","Confirmed","Deaths",
                                              "Recovered","Active","Combined_Key"))
        elif ind == 3:
            str_inds = [0,1,2,3,11]
            date_ind = 4
            int_inds = [7,8,9,10]
            float_inds = [5,6,12,13]
            parsed_lines = [parse_row(r.strip(),str_inds,date_ind,int_inds,float_inds) for r in lines]
            df_append = pd.DataFrame(parsed_lines,
                                     columns=("FIPS","County","Province/State",
                                              "Country/Region","Updated Date/Time",
                                              "Latitude","Longitude","Confirmed","Deaths",
                                              "Recovered","Active","Combined_Key","Incidence Rate","Case-Fatality_Ratio"))
        else:
            if l_local:
                loc = data_dir+date_str+".csv"
            else:
                loc = base_url.format(date_str)
            print("Structure of file from {0} is not recognized".format(loc))
            sys.exit(1)



        # Add date to which this was associated
        df_append["Date"] = loop_date
        df_append["N Days From Start"] = n_loop
        df = df.append(df_append,sort=False) 
        
        loop_date += datetime.timedelta(days=1)
        n_loop += 1

    # Make sure all FIPS are 5 digits
    df["FIPS"] = df["FIPS"].apply(make_5dig_FIPS)

    df = df[['FIPS','County','Province/State','Country/Region',
             'Date', 'Confirmed','Deaths','Combined_Key','N Days From Start']]

    return df


def parse_row(row,str_inds,date_ind,int_inds,float_inds):
    """
    Parse each line/row into a list with the appropriate data types
    """
    spamreader = csv.reader([row],delimiter=',',quotechar='"')
    vals = list(spamreader)[0]
    # Just gonna loop over all the vals here (prob slow but ehh...)
    for i in range(len(vals)):
        if i == date_ind:
            # They decided to have ADHD when it comes to date formats...
            for l in range(len(date_formats)):
                try:
                    vals[i] = datetime.datetime.strptime(vals[i],date_formats[l])
                    break
                except ValueError:
                    pass
            else: # only actived without a break statement
                print("No date format for '{0}'".format(vals[i]))
                sys.exit(1)
        elif i in int_inds:
            if vals[i] == "":
                vals[i] = 0
            else:
                vals[i] = int(vals[i])
        elif i in float_inds:
            if vals[i] == "":
                vals[i] = 0.0
            else:
                vals[i] = float(vals[i])
        else:
            assert i in str_inds
       
 
    return vals

def make_5dig_FIPS(FIPS):
    if type(FIPS)==str and len(FIPS) < 5:
        for i in range(5-len(FIPS)):
            FIPS = "0" + FIPS
    return FIPS

def get_plt_dfs(df):
    """
    Split the base dataframe into dataframes for the different regions (e.g. counties, states)
    """
    ### COUNTIES ###
    # Remove NaNs and blanks
    df_fips = df[(df["FIPS"].apply(lambda x : x == x and x != "")) &
                              (df["County"] != "") &
                              (df["Country/Region"] == "US")].copy()

    # Sort by FIPS and date
    df_fips = df_fips.sort_values(by=["FIPS","Date"]).reset_index(drop=True)

    # Get logscale values
    df_fips["Log10_Confirmed"] = df_fips["Confirmed"].apply(lambda x: np.log10(x))
    df_fips["Log10_Deaths"] = df_fips["Deaths"].apply(lambda x: np.log10(x))

    # New Confirmed & Deaths (maybe find a way to vectorize this?)
    df_fips["New_Confirmed"] = np.nan
    df_fips["New_Deaths"] = np.nan
    for i in range(1,len(df_fips)):
        if ((df_fips["FIPS"].iat[i] == df_fips["FIPS"].iat[i-1]) and
                (df_fips["County"].iat[i] == df_fips["County"].iat[i-1]) and
                (df_fips["Date"].iat[i] == df_fips["Date"].iat[i-1] + datetime.timedelta(days=1))):
            df_fips["New_Confirmed"].iat[i] = df_fips["Confirmed"].iat[i] - df_fips["Confirmed"].iat[i-1]
            df_fips["New_Deaths"].iat[i] = df_fips["Deaths"].iat[i] - df_fips["Deaths"].iat[i-1]

    ### Some specific calcs for New Confirmed and Deaths ### (So much improvement can be done here...)
    # End Date - get New Confirmed/Deaths for previous week & month (30 days)
    df_fips["Last_Week_Confirmed"] = np.nan
    df_fips["Last_Week_Deaths"] = np.nan
    for i in range(7,len(df_fips)):
        if ((df_fips["Date"].iat[i] == end_date) and
                (df_fips["County"].iat[i] == df_fips["County"].iat[i-7]) and
                (df_fips["Province/State"].iat[i] == df_fips["Province/State"].iat[i-7]) and
                (df_fips["Date"].iat[i] == df_fips["Date"].iat[i-7] + datetime.timedelta(days=7))):
            df_fips["Last_Week_Confirmed"].iat[i] = df_fips["Confirmed"].iat[i] - df_fips["Confirmed"].iat[i-7]
            df_fips["Last_Week_Deaths"].iat[i] = df_fips["Deaths"].iat[i] - df_fips["Deaths"].iat[i-7]
    # ... month (30 days)
    df_fips["Last_Month_Confirmed"] = np.nan
    df_fips["Last_Month_Deaths"] = np.nan
    for i in range(30,len(df_fips)):
        if ((df_fips["Date"].iat[i] == end_date) and
                (df_fips["County"].iat[i] == df_fips["County"].iat[i-30]) and
                (df_fips["Province/State"].iat[i] == df_fips["Province/State"].iat[i-30]) and
                (df_fips["Date"].iat[i] == df_fips["Date"].iat[i-30] + datetime.timedelta(days=30))):
            df_fips["Last_Month_Confirmed"].iat[i] = df_fips["Confirmed"].iat[i] - df_fips["Confirmed"].iat[i-30]
            df_fips["Last_Month_Deaths"].iat[i] = df_fips["Deaths"].iat[i] - df_fips["Deaths"].iat[i-30]
    # If at start of week - calc new Confirmed/Deaths over the following week
    # If at start of month - calc new Confirmed/Deaths over the following month
    df_fips["Calendar_Week_Confirmed"] = np.nan
    df_fips["Calendar_Week_Deaths"] = np.nan
    df_fips["Calendar_Month_Confirmed"] = np.nan
    df_fips["Calendar_Month_Deaths"] = np.nan
    for i in range(len(df_fips)):
        if (df_fips["Date"].iat[i].weekday() == 6): # Sunday (end of last week)
            # Find last weekday of that week
            i_end = 'Nope!'
            for j in range(i+1,i+9): # go past next Sunday
                if ((j == len(df_fips)) or
                        (df_fips["County"].iat[i] != df_fips["County"].iat[j]) or
                        (df_fips["Province/State"].iat[i] != df_fips["Province/State"].iat[j]) or 
                        (df_fips["Date"].iat[i+1].isocalendar()[1] != df_fips["Date"].iat[j].isocalendar()[1])):
                    i_end = j-1
                    break
            # Calc the number of new cases for the calendar week starting with the new cases on monday (and
            #       assign it to that Monday - i+1)
            if i_end > i:
                df_fips["Calendar_Week_Confirmed"].iat[i+1] = df_fips["Confirmed"].iat[i_end] - df_fips["Confirmed"].iat[i]
                df_fips["Calendar_Week_Deaths"].iat[i+1] = df_fips["Deaths"].iat[i_end] - df_fips["Deaths"].iat[i]
        if (df_fips["Date"].iat[i].day == 1): # start of month
            # Use a previous date or set to 0
            prev_zero = False
            if ((i == 0) or 
                (df_fips["County"].iat[i] != df_fips["County"].iat[i-1]) or 
                (df_fips["Province/State"].iat[i] != df_fips["Province/State"].iat[i-1]) or 
                (df_fips["Date"].iat[i] != df_fips["Date"].iat[i-1] + datetime.timedelta(days=1))):
                prev_zero = True
            i_end = 'Nope!'
            # Find last day of that month
            for j in range(i+1,i+32): # go to next Sunday
                if ((j >= len(df_fips)) or
                        (df_fips["County"].iat[i] != df_fips["County"].iat[j]) or
                        (df_fips["Province/State"].iat[i] != df_fips["Province/State"].iat[j]) or
                        (df_fips["Date"].iat[i].month != df_fips["Date"].iat[j].month)):
                    i_end = j-1
                    break
            if i_end > i:
                if prev_zero:
                    df_fips["Calendar_Month_Confirmed"].iat[i] = df_fips["Confirmed"].iat[i_end]
                    df_fips["Calendar_Month_Deaths"].iat[i] = df_fips["Deaths"].iat[i_end]
                else:
                    df_fips["Calendar_Month_Confirmed"].iat[i] = df_fips["Confirmed"].iat[i_end] - df_fips["Confirmed"].iat[i-1]
                    df_fips["Calendar_Month_Deaths"].iat[i] = df_fips["Deaths"].iat[i_end] - df_fips["Deaths"].iat[i-1]

    # No idea why but if I don't write this to a csv first and then reload it, it takes forever to run in dash...
    df_fips.to_csv("./df_counties.csv",index=False)   
    df_county = pd.read_csv("./df_counties.csv",dtype={"FIPS": str})
      

    ### STATES ###
    df_states = df[(df["Country/Region"] == "US") &
                    df["Province/State"].apply(lambda x : x == x and x != "")].copy()

    # Join by state and date
    df_state = df_states.groupby(['Date','Province/State',"N Days From Start"]).agg(
                                                    {"Confirmed":np.nansum,
                                                    "Deaths": np.nansum}).reset_index()

    # Sort by State and date
    df_state = df_state.sort_values(by=["Province/State","Date"]).reset_index(drop=True)

    df_state["Log10_Confirmed"] = df_state["Confirmed"].apply(lambda x: np.log10(x))
    df_state["Log10_Deaths"] = df_state["Deaths"].apply(lambda x: np.log10(x))

    # New Confirmed & Deaths (maybe find a way to vectorize this?)
    df_state["New_Confirmed"] = np.nan
    df_state["New_Deaths"] = np.nan
    for i in range(1,len(df_state)):
        if ((df_state["Province/State"].iat[i] == df_state["Province/State"].iat[i-1]) and
                (df_state["Date"].iat[i] == df_state["Date"].iat[i-1] + datetime.timedelta(days=1))):
            df_state["New_Confirmed"].iat[i] = df_state["Confirmed"].iat[i] - df_state["Confirmed"].iat[i-1]
            df_state["New_Deaths"].iat[i] = df_state["Deaths"].iat[i] - df_state["Deaths"].iat[i-1]

    ### Some specific calcs for New Confirmed and Deaths ### (So much improvement can be done here...)
    # End Date - get New Confirmed/Deaths for previous week & month (30 days)
    df_state["Last_Week_Confirmed"] = np.nan
    df_state["Last_Week_Deaths"] = np.nan
    for i in range(7,len(df_state)):
        if ((df_state["Date"].iat[i] == end_date) and
                (df_state["Province/State"].iat[i] == df_state["Province/State"].iat[i-7]) and
                (df_state["Date"].iat[i] == df_state["Date"].iat[i-7] + datetime.timedelta(days=7))):
            df_state["Last_Week_Confirmed"].iat[i] = df_state["Confirmed"].iat[i] - df_state["Confirmed"].iat[i-7]
            df_state["Last_Week_Deaths"].iat[i] = df_state["Deaths"].iat[i] - df_state["Deaths"].iat[i-7]
    # ... month (30 days)
    df_state["Last_Month_Confirmed"] = np.nan
    df_state["Last_Month_Deaths"] = np.nan
    for i in range(30,len(df_state)):
        if ((df_state["Date"].iat[i] == end_date) and
                (df_state["Province/State"].iat[i] == df_state["Province/State"].iat[i-30]) and
                (df_state["Date"].iat[i] == df_state["Date"].iat[i-30] + datetime.timedelta(days=30))):
            df_state["Last_Month_Confirmed"].iat[i] = df_state["Confirmed"].iat[i] - df_state["Confirmed"].iat[i-30]
            df_state["Last_Month_Deaths"].iat[i] = df_state["Deaths"].iat[i] - df_state["Deaths"].iat[i-30]
    # If at start of week - calc new Confirmed/Deaths over the following week
    # If at start of month - calc new Confirmed/Deaths over the following month
    df_state["Calendar_Week_Confirmed"] = np.nan
    df_state["Calendar_Week_Deaths"] = np.nan
    df_state["Calendar_Month_Confirmed"] = np.nan
    df_state["Calendar_Month_Deaths"] = np.nan
    for i in range(len(df_state)):
        if (df_state["Date"].iat[i].weekday() == 6): # Sunday (end of last week)
            # Find last weekday of that week
            i_end = 'Nope!'
            for j in range(i+1,i+9): # go past next Sunday
                if ((j == len(df_state)) or
                        (df_state["Province/State"].iat[i] != df_state["Province/State"].iat[j]) or 
                        (df_state["Date"].iat[i+1].isocalendar()[1] != df_state["Date"].iat[j].isocalendar()[1])):
                    i_end = j-1
                    break
            # Calc the number of new cases for the calendar week starting with the new cases on monday (and
            #       assign it to that Monday - i+1)
            if i_end > i:
                df_state["Calendar_Week_Confirmed"].iat[i+1] = df_state["Confirmed"].iat[i_end] - df_state["Confirmed"].iat[i]
                df_state["Calendar_Week_Deaths"].iat[i+1] = df_state["Deaths"].iat[i_end] - df_state["Deaths"].iat[i]
        if (df_state["Date"].iat[i].day == 1): # start of month
            # Use a previous date or set to 0
            prev_zero = False
            if ((i == 0) or 
                (df_state["Province/State"].iat[i] != df_state["Province/State"].iat[i-1]) or 
                (df_state["Date"].iat[i] != df_state["Date"].iat[i-1] + datetime.timedelta(days=1))):
                prev_zero = True
            i_end = 'Nope!'
            # Find last day of that month
            for j in range(i+1,i+32): # go to next Sunday
                if ((j >= len(df_state)) or
                        (df_state["Province/State"].iat[i] != df_state["Province/State"].iat[j]) or
                        (df_state["Date"].iat[i].month != df_state["Date"].iat[j].month)):
                    i_end = j-1
                    break
            if i_end > i:
                if prev_zero:
                    df_state["Calendar_Month_Confirmed"].iat[i] = df_state["Confirmed"].iat[i_end]
                    df_state["Calendar_Month_Deaths"].iat[i] = df_state["Deaths"].iat[i_end]
                else:
                    df_state["Calendar_Month_Confirmed"].iat[i] = df_state["Confirmed"].iat[i_end] - df_state["Confirmed"].iat[i-1]
                    df_state["Calendar_Month_Deaths"].iat[i] = df_state["Deaths"].iat[i_end] - df_state["Deaths"].iat[i-1]

    return df_county, df_state

#################### Executables ##############################################
t0 = time.time() 

# Create Data dir
init()

# Get foundational dataframe
df = get_data()

# Get regional dataframes
df_county, df_state = get_plt_dfs(df)

# Pickle the dataframes to save them for the next run on the same day
today = datetime.date.today().strftime("%Y_%m_%d")
df_county.to_pickle(base_county_pickle_name.format(today))
df_state.to_pickle(base_state_pickle_name.format(today))


te = time.time()
print("Execution Time: {0:5.2f}s".format(te-t0))



# Old Code: Doubling Rates
# Counties
#    # Calculate doubling rate from the last week
#    df_fips["Confirmed Doubling Rate"] = np.nan
#    df_fips["Death Doubling Rate"]     = np.nan
#    for i in range(len(df_fips)):
#        if df_fips["Date"].iat[i] == end_date:

#            # Find last 7 days for that FIPS (including end date)
#            df_dat =  df_fips.loc[(df_fips["FIPS"].iat[i] == df_fips["FIPS"]) & ((df_fips["N Days From Start"].iat[i] - 
#                                  df_fips["N Days From Start"]) <= 6)]
#            if len(df_dat) == 7:
#                # Confirmed
#                y_dat = df_dat["Confirmed"]
#                if y_dat.iloc[0] > 0 and y_dat.is_monotonic_increasing and y_dat.iloc[0] < y_dat.iloc[-1]: # Must be increasing for this to be valid
#                    popt, pcov = curve_fit(lambda t,a,b: a*np.exp(b*t), np.arange(7), y_dat, p0=(y_dat.iloc[0],.1)) # 0.1 approx. 5 day doubling rate
#                    df_fips["Confirmed Doubling Rate"].iat[i] = np.log(2)/popt[1]
#                # Deaths
#                y_dat = df_dat["Deaths"]
#                if y_dat.iloc[0] > 0 and y_dat.is_monotonic_increasing and y_dat.iloc[0] < y_dat.iloc[-1]: # Must be increasing for this to be valid
#                    popt, pcov = curve_fit(lambda t,a,b: a*np.exp(b*t), np.arange(7), y_dat, p0=(y_dat.iloc[0],.1)) # 0.1 approx. 5 day doubling rate
#                    df_fips["Death Doubling Rate"].iat[i] = np.log(2)/popt[1]
## States
#    # Calculate doubling rate from the last week
#    df_state_merge["Confirmed Doubling Rate"] = np.nan
#    df_state_merge["Death Doubling Rate"]     = np.nan
#    for i in range(len(df_state_merge)):
#        if df_state_merge["Date"].iat[i] == end_date:

#            # Find last 7 days for that State (including end date)
#            df_dat =  df_state_merge.loc[(df_state_merge["Province/State"].iat[i] == df_state_merge["Province/State"]) & 
#                                         ((df_state_merge["N Days From Start"].iat[i] - 
#                                           df_state_merge["N Days From Start"]) <= 6)]
#            if len(df_dat) == 7:
#                # Confirmed
#                y_dat = df_dat["Confirmed"]
#                if y_dat.iloc[0] > 0 and y_dat.is_monotonic_increasing and y_dat.iloc[0] < y_dat.iloc[-1]: # Must be increasing for this to be valid
#                    popt, pcov = curve_fit(lambda t,a,b: a*np.exp(b*t), np.arange(7), y_dat, p0=(y_dat.iloc[0],.1)) # 0.1 approx. 5 day doubling rate
#                    df_state_merge["Confirmed Doubling Rate"].iat[i] = np.log(2)/popt[1]
#                # Deaths
#                y_dat = df_dat["Deaths"]
#                if y_dat.iloc[0] > 0 and y_dat.is_monotonic_increasing and y_dat.iloc[0] < y_dat.iloc[-1]: # Must be increasing for this to be valid
#                    popt, pcov = curve_fit(lambda t,a,b: a*np.exp(b*t), np.arange(7), y_dat, p0=(y_dat.iloc[0],.1)) # 0.1 approx. 5 day doubling rate
#                    df_state_merge["Death Doubling Rate"].iat[i] = np.log(2)/popt[1]

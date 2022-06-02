import numpy as np
import pandas as pd


# clean the input Excel spreadsheet and return a clean df
def cleanup(filepath):
    print(filepath)
    # import data and transpose
    df_raw = pd.read_excel(filepath, header=None).T

    # modify the header
    new_header = df_raw.iloc[0]  # grab the first row for the header
    df = df_raw[1:]  # take the data except the header row
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)

    # add new column for visualization
    df['Start Datetime'] = pd.to_datetime(df['Start Date'] + ' ' + df['Start Time'])
    df['End Datetime'] = pd.to_datetime(df['End Date'] + ' ' + df['End Time'])
    df['Timeout'] = df['Active Lever Presses'] - df['Reward']

    # drop unnecessary columns
    df.drop(['Start Date', 'Start Time', 'End Date', 'End Time'], axis=1, inplace=True)

    # change data types
    cols = df.columns.tolist()
    for col in cols:
        name = col.lower()
        if ('active' in name) or ('reward' in name) or ('timeout' in name):
            df[col] = df[col].astype('int32')
            df[col] = df[col].replace(0, np.nan)
        else:
            pass

    # drop columns which all values are Nan
    df.dropna(how='all', axis=1, inplace=True)

    # reorder the columns
    new_columns = df.columns.tolist()
    new_columns.insert(0, new_columns.pop(new_columns.index('Subject')))
    new_columns.insert(1, new_columns.pop(new_columns.index('Start Datetime')))
    new_columns.insert(2, new_columns.pop(new_columns.index('End Datetime')))

    idx = new_columns.index('Reward') + 1
    new_columns.insert(idx, new_columns.pop(new_columns.index('Timeout')))

    # fill the nan with 0
    df = df[new_columns]
    df.fillna(0, inplace=True)
    print('CLEANING COMPLETED')
    # output the result dataframe
    return df


# return a df that contains reward latency, all rewards, intervals, and cleaned intervals
def filtered_reward(df):
    filtered_cols = ['Subject'] + [col for col in df.columns if 'Reward ' in col]
    df_reward = df[filtered_cols].sort_values('Subject').reset_index().drop('index', axis=1)
    df_reward['allRewards'] = df_reward.iloc[:, 1:].values.tolist()
    df_reward['Latency'] = df_reward['Reward 1']
    df_filtered = df_reward[['Subject', 'Latency', 'allRewards']]
    df_filtered['Intervals'] = df_filtered['allRewards'].apply(lambda lst: [j - i for i, j in zip(lst[:-1], lst[1:])])
    df_filtered['cleanedIntervals'] = df_filtered['Intervals'].apply(lambda lst: [val for val in lst if val > 0][:-1])

    return df_filtered

import pandas as pd 
import glob 
import re
import numpy as np
from utils import rename_ease
from utils import join_columns


def collate_data(data_path):
    li = []
    for filename in data_path:
        df = pd.read_excel(filename, index_col = None, header = 0) 


        df['Date'] = pd.to_datetime(df['Date'], format = '%Y%m%d') #converting from a string to datetime 
        #removing unnecessary columns 
        df.drop(['URL', 'Device type', 'Browser', 'System', 'IP', 'Submitted', 'Battery', 'Network Connection', 'Orientation', 'App name', 'App SDK version', 'App version', 
        'ContentsquareReplay', 'policyNumber', 'Time'], axis = 1, inplace = True, errors = 'ignore') #errors set to ignore so that columns that only occur in one type of input don't raise an error and still get dropped 

        df.drop(list(df.filter(regex = 'translated')), axis = 1, inplace = True) #drop all columns with strings containing 'translated'

        df = rename_ease(df,data_path)

        li.append(df)  
        
    feedback_data = pd.concat(li, axis = 0, ignore_index = False)

    feedback_data.fillna("", inplace = True) #filling null values with blank space for concatenation 
    
    feedback_data = join_columns(feedback_data, data_path)

    feedback_data['all_comments']=feedback_data['all_comments'].map(lambda x: x.strip()) #strip leading and trailing whitespace 

    feedback_data['all_comments']=feedback_data['all_comments'].map(lambda x: re.sub(r'(x)\1{1,}','', x)) #replace all strings containing just x's with null values 

    feedback_data['all_comments']=feedback_data['all_comments'].map(lambda x: re.sub(r'^[a-zA-Z]{2}$','', x)) #dropping strings containing only two letters

    feedback_data['all_comments']=feedback_data['all_comments'].map(lambda x: re.sub(r'[,.!?]{2}', '', x)) #remove double punctuations

    feedback_data['all_comments']=feedback_data['all_comments'].map(lambda x: re.sub(r'^(\w)\1{2,}$', '', x, flags = re.IGNORECASE)) #remove repeating characters 

    feedback_data['all_comments']=feedback_data['all_comments'].map(lambda x: re.sub(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', '',x)) #email

    feedback_data['all_comments'].replace(r'\n',' ', regex=True, inplace=True)

    feedback_data['all_comments'].replace(r'^\s*$',np.nan, regex=True, inplace=True)


    #feedback_data['all_comments'].isna().sum() #checking wether white space has been transforemd 

    feedback_data['easy_hard'] = pd.cut(feedback_data.ease, bins = [0,3,5], labels= ['hard', 'easy']) #if ease is 4 or 5 deemed easy, 3 or under was hard

    print('The total number of rows in the dataset is {}'.format(len(feedback_data)))

    feedback_data.dropna(subset = ['all_comments'], inplace = True) #dropping all rows with no comments 

    print('The usable number of rows in the dataset is {}'.format(len(feedback_data)))

    #putting date column in chronological order and creating month_year column for ldaseq time_slice 
    feedback_data.reset_index(inplace =True)
    feedback_data = feedback_data.sort_values(by = ['Date'])
    feedback_data['month_year'] = pd.to_datetime(feedback_data['Date']).dt.to_period('M')
    feedback_data.drop(['index'], axis= 1)
    
    return feedback_data
    

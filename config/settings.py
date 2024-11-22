# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:56:23 2024

@author: AnatolFiete.Naeher
"""

path = ('J:\\5_Forschung\\5.6_Research overview\\Forschungsprojekte\\'
        'Representativity in Digital Real-World Trials\\'
        'Did DHT increase representativity in Clinical Trials\\Data')

param = {
    'dbname': 'aact',
    'user': "user\\",  
    'password': "password",   
    'host': 'aact-db.ctti-clinicaltrials.org',
    'port': '5432'
}

prompt_template = '''
You are provided with a labeling task: label the phrase {exp} with one and only 
one of the labels in the list {cat}. Output the result as a list:

['cat_title', 'cat_exp', 'piv_cat'],

where

'cat_title' = {title},
'cat_exp' = {exp}
and 'piv_cat' designates the assigned label from list {cat}.
'''

cat_dict = {
    'gender': ['male', 'female', 'other', 'unknown_g'],
    'age' : ['<18 years', 'between 18 and 65 years', '>65 years', 'unknown_a'],
    'race' : ['American Indian or Alaska Native', 'Asian',
              'Black or African American', 'Hispanic or Latino',
              'Native Hawaian or Other Pacific Islander', 'White', 'unknown_r']
}


openAI_Model = "gpt-4o-mini"
chunks = 5
ctdwn = 1

filename = 'CTD'
batch_filename = 'b_output'

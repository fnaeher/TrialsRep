# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:56:23 2024

@author: AnatolFiete.Naeher and Akhyar.Ahmed
"""

path = ('/Users/akhyar.ahmed/Downloads/Data')

param = {
    'dbname': 'aact',
    'user': "user/",  
    'password': "pass",   
    'host': 'aact-db.ctti-clinicaltrials.org',
    'port': '5432'
}

prompt_template = '''
You are provided with a labeling task:
- The phrase is "{exp}".
- The only valid labels are one of: "{cat}".
Output strictly valid JSON with no additional commentary, in the format:

{{
  "cat_title": "{title}",
  "cat_exp": "{exp}",
  "piv_cat": "<One label from {cat}>"
}}

Where:
- "cat_title" = {title},
- "cat_exp" = {exp},
- "piv_cat" is exactly one from the list {cat}.

Return only valid JSON in String. No disclaimers, no extra text.
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

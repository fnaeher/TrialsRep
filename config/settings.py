# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:56:23 2024

@author: AnatolFiete.Naeher
"""

path = ('path')

param = {
    'dbname': 'aact',
    'user': 'user',  
    'password': 'password',   
    'host': 'aact-db.ctti-clinicaltrials.org',
    'port': '5432'
}

filename = 'CTD'

urls = {
    'Map_1': 'https://raw.githubusercontent.com/pantapps/cbms2019/refs/heads/'\
    'master/mesh_icd10cm_via_snomedct_not_mapped_umls.tsv',

    'Map_2': 'https://raw.githubusercontent.com/pantapps/cbms2019/refs/heads/'\
    'master/mesh_snomedct_via_icd10cm_not_mapped_umls.tsv'
    }
    
f_list = {
    'list_1':[f"{filename}_1.csv", f"{filename}_2.csv", f"{filename}_3.csv", 
        f"{filename}_4.csv", f"{filename}_5.csv", f"{filename}_6.csv"],
    
    'list_2': ['DD.csv', 'u_flags.csv','b_output.csv','MICD.csv',\
        'dht_icd10.csv']
    }

openAI_Model = "gpt-4o-mini"
chunks = 5
ctdwn = 1
batch_filename = 'b_output'

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

dflg_filename = '_DHT_Flags'
dflg_colname = 'DHT'
dflg_columns = columns = ['official_title', 'name', 'q_3_desc', 'q_7_desc', 
                          'title', 'q_8_desc', 'measure', 'q_9_desc', 
                          'q_10_desc']

group = 'nct_id'

ICD10_descriptor = [
    ('A00', 'B99', 'Infectious diseases'),
    ('C00', 'D48', 'Neoplasms'),
    ('D50', 'D90', 'Hematology/Immunology'),
    ('E00', 'E90', 'Endocrine, nutritional and metabolic diseases'),
    ('F00', 'F99', 'Mental Health'),
    ('G00', 'G99', 'Nervous system'),
    ('H00', 'H59', 'Eye and Adnexa'),
    ('H60', 'H95', 'Ear and Mastoid Process'),
    ('I00', 'I99', 'Circulatory System'),
    ('J00', 'J99', 'Respiratory System'),
    ('K00', 'K93', 'Digestive System'),
    ('L00', 'L99', 'Skin and Subcutaneous Tissue'),
    ('M00', 'M99', 'Musculoskeletal System and Connective Tissue'),
    ('N00', 'N99', 'Genitourinary System'),
    ('O00', 'O99', 'Pregnancy and Childbirth'),
    ('P00', 'P96', 'Conditions during Perinatal Period'),
    ('Q00', 'Q99', 'Congenital Malformations, Deformations and Chromosomal' 
                     'Abnormalities'),
    ('R00', 'R99', 'Symptoms, Signs and Abnormal Clinical and Laboratory' 
                     'Findings - not elsewhere classified'),
    ('S00', 'T98', 'Injury, Poisoning and other External Causes'),
    ('V01', 'Y84', 'External Causes of Morbidity and Mortality'),
    ('Z00', 'Z99', 'Factors influencing Health Status and Contact with Health' 
                     'Services'),
    ('U00', 'U99', 'Special purposes')
]

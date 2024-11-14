# DHTTrialsRep

DHTTrialsRep maps unstructured sociodemographic information from clinical trials registered on clinicaltrials.gov to predefined sociodemographic categories using batch_prompts via openAI's chat.completions API endpoint. Current work in preparation includes mapping of MeSH terms indcluded in clinicaltrials.gov data to ICD codes using the UMLS metathesaurus and the development of scores gauging mapping quality.  

Besides sociocdemographic information on clinical trials, the output also includes US CENSUS data for sosciodemographic represantivity comparisons and preliminary ICD mappings.

# Setting up DHTTrialsRep
(1) Clone the repository to your local drive and ensure that its folder structure is kept.  
(2) Create a folder for data storage and copy the csv-files included in the data.  
(3) Specify the data folder's path in 'settings.py'. You can also set the number of batch job chunks to process and the time interval at which to check for batch job completions. It is further required to specify names for the main and batch job output files.  
(4) Run 'trials.py'. 

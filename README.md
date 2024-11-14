# TrialsRep

TrialsRep maps unstructured sociodemographic information from clinical trials registered on clinicaltrials.gov to predefined sociodemographic categories using batch prompts via openAI's chat.completions API endpoint. Current work in preparation includes mapping of trials utilizing Digital Health Technologies (DHT) to the DHT usage categories as defined in Marra, C. et al. (2020) as well as mapping of MeSH terms indcluded in clinicaltrials.gov data to ICD codes using the UMLS metathesaurus and the development of scores gauging mapping quality.  

Besides sociodemographic information on clinical trials, the output also includes US CENSUS data for sociodemographic represantivity comparisons and preliminary ICD mappings. The following data is merged to clinicaltrials.gov data:  

- DD.csv -> US CENSUS data  
- DHTx.csv -> Preliminary data identifying DHT trials as defined by the FDA (e.g. Marra, C. et al. (2020))  
- MIC.csv -> Preliminary mappings of ICD codes  

# Setting up TrialsRep
(1) Register at https://aact.ctti-clinicaltrials.org/, get & implement an openAI client key.  
(2) Clone the repository to your local drive and ensure that its folder structure is kept.  
(3) Create a folder for data storage and copy the *.csv-files included in this repo's data folder.  
(4) Specify the data folder's path in 'settings.py'. You can also set the number of batch job chunks to be processed and the time interval at which to check for batch job completions. It is further required to specify names for the main and batch job output files. Enter your aact credentials.  
(5) Run 'trials.py'.

# References
Marra, C., Chen, J. L., Coravos, A., & Stern, A. D. (2020). Quantifying the use of connected digital products in clinical research. NPJ digital medicine, 3(1), 50.


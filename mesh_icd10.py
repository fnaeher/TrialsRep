import requests
import pandas as pd
import time

from datetime import datetime

UMLS_APIKEY = "6f7e6703-d59e-4c52-a56c-7eb448fe682c"
AUTH_ENDPOINT = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
SEARCH_ENDPOINT = "https://uts-ws.nlm.nih.gov/rest/search/current"
CONTENT_ENDPOINT = "https://uts-ws.nlm.nih.gov/rest/content/current"

def get_tgt(apikey):
    """
    Obtain a Ticket-Granting Ticket (TGT) from the UMLS API.
    """
    params = {'apikey': apikey}
    r = requests.post(AUTH_ENDPOINT, data=params)
    r.raise_for_status()
    # The TGT is provided in the 'location' header of the response
    return r.headers['location']

def get_service_ticket(tgt):
    """
    Exchange the TGT for a service ticket, used for each subsequent request.
    """
    params = {'service': 'http://umlsks.nlm.nih.gov'}
    r = requests.post(tgt, data=params)
    r.raise_for_status()
    return r.text

def get_cui_for_mesh_term(mesh_term, tgt):
    """
    Find the UMLS CUI for a given MeSH term using the UMLS search endpoint.
    """
    ticket = get_service_ticket(tgt)
    params = {
        'string': mesh_term,
        'ticket': ticket,
        'searchType': 'words',  # using 'words' for a more flexible search
        'sabs': 'MSH'
    }
    r = requests.get(SEARCH_ENDPOINT, params=params)
    r.raise_for_status()
    data = r.json()

    results = data['result']['results']
    if not results:
        print(f"No results found for MeSH term: {mesh_term}")
        return None

    # Take the first valid result's CUI (could be improved by more sophisticated selection)
    for result in results:
        ui = result['ui']
        if ui != "NONE":
            print(f"Found CUI: {ui} for MeSH term: {mesh_term}")
            return ui
    print(f"No valid CUI found for MeSH term: {mesh_term}")
    return None

def get_icd10_codes_for_cui(cui, tgt, year_threshold=None):
    """
    Given a UMLS CUI, retrieve associated ICD-10 codes.
    If a year_threshold is provided, only include atoms updated in or after that year.
    """
    ticket = get_service_ticket(tgt)
    url = f"{CONTENT_ENDPOINT}/CUI/{cui}/atoms"
    params = {'ticket': ticket}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    icd10_codes = []
    for atom in data['result']:
        # Use startswith to match ICD10 or ICD10CM sources
        if atom['rootSource'].startswith('ICD10') or 'ICD10ENG' in atom['rootSource']:
            # Optional date filtering: if the atom has an "updated" field, and a threshold is provided
            if year_threshold and "updated" in atom:
                try:
                    # Assume the date is in a format where the first 4 characters are the year
                    updated_year = int(atom["updated"][:4])
                except ValueError:
                    updated_year = None
                if updated_year and updated_year < year_threshold:
                    continue  # skip atoms not updated in the last X years
            code = atom['code']
            if code not in icd10_codes:
                icd10_codes.append(code)
    return icd10_codes

def process_mesh_terms(mesh_terms, years=10):
    """
    Process a list of MeSH terms and retrieve their CUI and ICD-10 codes.
    Optionally, only include ICD-10 atoms updated within the last 'years' years.
    """
    current_year = datetime.now().year
    year_threshold = current_year - years
    tgt = get_tgt(UMLS_APIKEY)
    results = {}
    for term in mesh_terms:
        cui = get_cui_for_mesh_term(term, tgt)
        if cui:
            icd10_codes = get_icd10_codes_for_cui(cui, tgt, year_threshold=year_threshold)
            results[term] = {"CUI": cui, "ICD10_codes": icd10_codes}
        else:
            results[term] = {"CUI": None, "ICD10_codes": []}
    return results

def main():
    # Option 1: Define a static list of MeSH terms
    # mesh_terms = ["Glucose Metabolism Disorders", "Pain", "Nutrition Disorders"]
    df = pd.read_csv('Data/CTD_5.csv', encoding='UTF-8')
    for idx, row in df.iterrows():
        # Option 2: Read a comma-separated list from user input (dynamic)
        # user_input = input("Enter comma-separated MeSH terms: ")
        user_input = row['mesh_term']
        mesh_terms = [term.strip() for term in user_input.split(",") if term.strip()]

        results = process_mesh_terms(mesh_terms, years=10)
        for term, info in results.items():
            code_ls = []
            for code_url in info['ICD10_codes']:
                code_ls.append(code_url.split('/')[-1])
            print(f"MeSH term: {term}")
            print(f"  CUI: {info['CUI']}")
            print(f"  ICD-10 codes: {code_ls}")
            print("-----")
        time.sleep(1)

if __name__ == "__main__":
    main()

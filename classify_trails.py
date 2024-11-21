'''
Created on THR Nov 21 11:43:13 2024

Pre-requisite: 
1. pip install pandas spacy
2. python -m spacy download en_core_web_sm

To run it:
python -m classify_trails
'''

import pandas as pd
import spacy
import logging
from collections import Counter


# Classifier class
class TrialClassifier:
    def __init__(self, data_file):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.data_file = data_file
        self.nlp = spacy.load('en_core_web_sm')
        self.df = None
        self.product_flag_columns = None
        self.class_keywords = None

    def load_data(self):
        logging.info('Step 1: Loading the dataset.')
        try:
            # Adjust the file loading method based on your file format
            if self.data_file.endswith('.csv'):
                self.df = pd.read_csv(self.data_file)
            elif self.data_file.endswith('.dta'):
                self.df = pd.read_stata(self.data_file)
            else:
                raise ValueError('Unsupported file format.')
            logging.info('Dataset loaded successfully.')
        except Exception as e:
            logging.error(f'Error loading dataset: {e}')
            exit()

    def extract_products_and_text(self):
        logging.info('Step 2: Extracting product flags and relevant text fields.')
        
        # Product flags start from column 40, so we will take all the features from the 40th columns
        self.product_flag_columns = self.df.columns[39:]

        # Function to extract products used in each trial
        def get_products_used(row):
            products = []
            for col in self.product_flag_columns:
                if row[col] == 1:
                    products.append(col)
            return products

        self.df['products_used'] = self.df.apply(get_products_used, axis=1)

        text_fields = ['primary_measure', 'secondary_measure', 'other_measure',
                       'primary_description', 'secondary_description', 'other_description',
                       'detailed_description', 'brief_summary', 'interventions']

        existing_text_fields = [field for field in text_fields if field in self.df.columns]

        def combine_text_fields(row):
            combined_text = ' '.join([str(row[field]) for field in existing_text_fields if pd.notnull(row[field])])
            return combined_text.lower()

        self.df['combined_text'] = self.df.apply(combine_text_fields, axis=1)
        logging.info('Product flags and text fields extracted and combined.')

    def define_keywords(self):
        logging.info('Step 3: Defining keywords for each class.')

        self.class_keywords = {
            'Class 1': ['validate', 'validation', 'equivalent', 'compare', 'comparison', 'accuracy', 'precision',
                        'measurement', 'sensitivity', 'specificity', 'reliability', 'agreement', 'concordance',
                        'reproducibility', 'performance'],
            'Class 2': ['usability', 'safety', 'tolerability', 'comfort', 'acceptability', 'patient engagement',
                        'retention', 'adherence', 'compliance', 'cost-effectiveness', 'satisfaction', 'experience',
                        'feasibility'],
            'Class 3': ['monitor', 'measure', 'assess', 'record', 'capture data', 'collect data', 'track', 'evaluate',
                        'observe'],
            'Class 4': ['intervention', 'treatment', 'therapy', 'efficacy', 'effectiveness', 'improve', 'reduce',
                        'increase', 'enhance', 'manage', 'alleviate', 'digital therapeutic']
        }

        # Basic text normalization
        for key in self.class_keywords:
            self.class_keywords[key] = [kw.lower() for kw in self.class_keywords[key]]

    def classify_trials(self):
        logging.info('Step 4: Analyzing text and classifying trials.')

        def classify_trial(row):
            classes_assigned = set()
            combined_text = row['combined_text']

            # Check for Class 2 keywords (can be combined with other classes)
            if any(kw in combined_text for kw in self.class_keywords['Class 2']):
                classes_assigned.add('Class 2')

            # For each product used in the trial
            for product in row['products_used']:
                product_classes = set()
                # For Classes 1, 3, and 4 (mutually exclusive per product)
                for cls in ['Class 1', 'Class 3', 'Class 4']:
                    if any(kw in combined_text for kw in self.class_keywords[cls]):
                        product_classes.add(cls)
                        break  # Assign the first matching class (mutually exclusive)
                if product_classes:
                    classes_assigned.update(product_classes)

            return list(classes_assigned)

        self.df['classes_assigned'] = self.df.apply(classify_trial, axis=1)
        logging.info('Trials classified successfully.')

    def save_results(self, output_file):
        logging.info('Step 5: Saving the classification results.')
        self.df.to_csv(output_file, index=False)
        logging.info(f'Classification results saved to {output_file}.')

    def print_summary(self):
        logging.info('Step 6: Printing a summary of classifications.')
        class_counts = Counter()
        for classes in self.df['classes_assigned']:
            class_counts.update(classes)

        print('\nSummary of Classifications:')
        for cls, count in class_counts.items():
            print(f'{cls}: {count} trials')
        logging.info('Processing completed.')

def main():
    data_file = './Data/device_trials_data_10Dec2019.dta'
    output_file = './Data/device_trials_data_10Dec2019_classified_trials.csv'

    # Initialize the classifier
    classifier = TrialClassifier(data_file)

    # Run the classification steps
    classifier.load_data()
    classifier.extract_products_and_text()
    classifier.define_keywords()
    classifier.classify_trials()
    classifier.save_results(output_file)
    classifier.print_summary()

# The main method
if __name__ == '__main__':
    main()

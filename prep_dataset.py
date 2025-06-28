import pandas as pd
import os
import re

# The new, correct filename
CSV_FILE = "Training.csv"
OUTPUT_DIR = "knowledge_base"

def create_knowledge_base_from_csv():
    """Reads the Kaggle Training.csv and creates a text file for each disease."""
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: The dataset file '{CSV_FILE}' was not found.")
        print("Please download it from https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning, and place 'Training.csv' in the project root.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Cleaning old files from '{OUTPUT_DIR}'...")
    for filename in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, filename))

    print(f"Loading dataset from '{CSV_FILE}'...")
    df = pd.read_csv(CSV_FILE)

    # The last column 'prognosis' is our target disease
    disease_col = 'prognosis'
    
    # Get a list of all unique diseases
    diseases = df[disease_col].unique()
    
    # Get a list of all symptom columns (all columns except the last one)
    # The dataset might have a trailing empty column, so we check for 'Unnamed'
    symptom_cols = [col for col in df.columns if col != disease_col and 'Unnamed' not in col]

    print(f"Found {len(diseases)} diseases and {len(symptom_cols)} symptoms.")

    # Process each unique disease
    for disease in diseases:
        # Create a clean filename like 'fungal_infection.txt'
        filename_disease = re.sub(r'[^a-z0-9_]', '', disease.lower().replace(' ', '_'))
        file_path = os.path.join(OUTPUT_DIR, f"{filename_disease}.txt")

        # Find the first row for this disease to get its symptoms
        disease_rows = df[df[disease_col] == disease]
        if disease_rows.empty:
            continue
        
        # Take the first profile for that disease
        disease_profile = disease_rows.iloc[0]

        # Find all symptoms that are marked with '1' for this disease profile
        present_symptoms = [
            col.replace('_', ' ') for col in symptom_cols if disease_profile[col] == 1
        ]
        
        # Assemble the content for the text file
        content = f"Information about: {disease}\n\n"
        content += "Key Symptoms:\n"
        for symp in present_symptoms:
            content += f"- {symp.strip()}\n"
        
        # Write the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"Successfully created {len(diseases)} knowledge files in the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    create_knowledge_base_from_csv()
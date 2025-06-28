import pandas as pd
import os

def create_knowledge_base():
    if not os.path.exists('Training.csv'):
        print("Training.csv not found. Please download the dataset first.")
        return
    
    df = pd.read_csv('Training.csv')
    
    if not os.path.exists('knowledge_base'):
        os.makedirs('knowledge_base')
    
    for index, row in df.iterrows():
        disease = row['prognosis']
        symptoms = row.drop('prognosis').to_dict()
        
        filename = f"knowledge_base/{disease.lower().replace(' ', '_')}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Information about: {disease}\n\n")
            f.write("Key Symptoms:\n")
            
            for symptom, value in symptoms.items():
                if value == 1:
                    f.write(f"- {symptom}\n")
            
            f.write(f"\nDisease: {disease}\n")
            f.write("This is a medical condition that may require professional diagnosis and treatment.\n")
            f.write("Please consult with a healthcare provider for proper medical advice.\n")
    
    print(f"Knowledge base created with {len(df)} disease files.")

if __name__ == "__main__":
    create_knowledge_base()
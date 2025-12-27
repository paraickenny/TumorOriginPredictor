import pandas as pd
import pickle

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 275)

# Load alias data
alias_df = pd.read_csv("Alias.csv")
main_gene_dict = {}

for _, row in alias_df.iterrows():
    main_gene = row["Gene"].strip()
    aliases = [alias.strip() for alias in row["Alias"].split(', ')]
    main_gene_dict[main_gene] = aliases

# Load MSK-IMPACT dataset
msk_impact_data = pd.read_csv("MSK-IMPACT-merged-binarized_338.txt", sep="\t")
msk_genes = sorted(msk_impact_data.columns[2:])

# Load patient data
patient_df = pd.read_csv("gene_names.txt", sep="\t")

# Function to map genes to valid aliases
def map_to_valid_aliases(main_gene, alias_dict, msk_genes):
    if main_gene not in alias_dict:
        return None
    
    for alias in alias_dict[main_gene]:
        if alias in msk_genes:
            return alias
    
    return None

# Process mutations
processed_mutations = []

for _, row in patient_df.iterrows():
    mutations = [gene.strip() for gene in row["Somatic_Mutations"].split(', ')]
    valid_genes = []
    
    for gene in mutations:
        if gene in msk_genes:
            valid_genes.append(gene)
        else:
            valid_alias = map_to_valid_aliases(gene, main_gene_dict, msk_genes)
            if valid_alias:
                valid_genes.append(valid_alias)
    
    processed_mutations.append(", ".join(valid_genes))

patient_df["Processed_Mutations"] = processed_mutations

# Define models
models = {
    "KNN": "knn.pkl",
    "Decision Tree": "dec_tree.pkl",
    "Logistic Regression": "logreg.pkl",
    "SVM": "svm.pkl",
    "Skmultilearn": "skm.pkl"
}

# Initialize DataFrame with appropriate columns
columns = ["Patient_ID", "True_Tissue_Origin", "Input_Genes"]
for model_name in models.keys():
    columns += [f"{model_name}_Top1", f"{model_name}_Top2", f"{model_name}_Top3"]

predictions_df = pd.DataFrame(columns=columns)

# Iterate over each patient
for _, row in patient_df.iterrows():
    patient_id = row["Study_ID"]
    true_tissue_origin = row["Origin_Tissue"]
    patient_genes = row["Processed_Mutations"].split(', ')
    
    unknowndict = {gene: 1 if gene in patient_genes else 0 for gene in msk_genes}
    unknown = pd.DataFrame([unknowndict])
    
    patient_predictions = {
        "Patient_ID": patient_id,
        "True_Tissue_Origin": true_tissue_origin,
        "Input_Genes": ", ".join(patient_genes)
    }
    
    for model_name, model_file in models.items():
        with open(model_file, 'rb') as f:
            model = pickle.load(f)

            # Ensure feature order matches training set
            if hasattr(model, "feature_names_in_"):
                unknown = unknown.reindex(columns=model.feature_names_in_, fill_value=0)

            # Get the top 3 predictions
            probabilities = model.predict_proba(unknown)[0]
            top_indices = probabilities.argsort()[-3:][::-1]
            top_predictions = [model.classes_[i] for i in top_indices]

            # Assign the top predictions to the corresponding columns
            patient_predictions[f"{model_name}_Top1"] = top_predictions[0]
            patient_predictions[f"{model_name}_Top2"] = top_predictions[1]
            patient_predictions[f"{model_name}_Top3"] = top_predictions[2]

    predictions_df = pd.concat([
        predictions_df,
        pd.DataFrame([patient_predictions])
    ], ignore_index=True)

# Save the predictions to a CSV file
predictions_df.to_csv("top_3_2-25-2024.csv", index=False)

print("\nPredictions for All Models have been saved to 'top_3.csv'")

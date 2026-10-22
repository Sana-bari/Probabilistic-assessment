import pandas as pd

# Path to the original Excel file containing all three sheets
file_path = "data/My Data Excel.xlsx"

# Load the specific sheets for Experimental, Zernike Moments, and Pseudo-Zernike Moments
sheets = pd.read_excel(file_path, sheet_name=["Experimental", "Zernike Moments", "Pseudo-Zernike Moments"], engine='openpyxl')

# Define a function to convert a sheet to binary using a specified threshold percentage
def convert_to_binary(df, threshold_percentage):
    # Keep the 'Taxon', 'Strain ID', and 'Colony ID' columns intact
    df_intact = df.iloc[:, :3]  # Columns A, B, C (1, 2, 3) intact
    
    # Subset the dataframe for binary conversion columns (moments)
    df_subset = df.iloc[:, 3:]
    
    # Apply threshold and convert to binary
    for col in df_subset.columns:
        max_value = df_subset[col].max()
        threshold = threshold_percentage * max_value
        df_subset[col] = (df_subset[col] >= threshold).astype(int)
    
    # Combine the intact and binary-converted columns
    df_combined = pd.concat([df_intact, df_subset], axis=1)
    
    return df_combined

# Convert each sheet using a 50% threshold for Experimental, Zernike, and Pseudo-Zernike data
df_experimental_50pct = convert_to_binary(sheets["Experimental"], threshold_percentage=0.5)
df_zernike_50pct = convert_to_binary(sheets["Zernike Moments"], threshold_percentage=0.5)
df_pseudozernike_50pct = convert_to_binary(sheets["Pseudo-Zernike Moments"], threshold_percentage=0.5)

# Save the binary-converted files to the specified output location
experimental_output_file = "results/50%_Experimental_Binary_Conversion.xlsx"
zernike_output_file = "results/50%_Zernike_Moments_Binary_Conversion.xlsx"
pseudozernike_output_file = "results/50%_Pseudo-Zernike_Moments_Binary_Conversion.xlsx"

df_experimental_50pct.to_excel(experimental_output_file, engine='openpyxl', index=False)
df_zernike_50pct.to_excel(zernike_output_file, engine='openpyxl', index=False)
df_pseudozernike_50pct.to_excel(pseudozernike_output_file, engine='openpyxl', index=False)

# Calculate the probability matrix for each binary-converted dataframe
species_column = 'Taxon'

# Group by species and calculate the percentage of 1s for each column in Experimental
experimental_percentage = df_experimental_50pct.groupby(species_column).apply(lambda x: (x.iloc[:, 3:].sum() / len(x)) * 100)

# Group by species and calculate the percentage of 1s for each column in Zernike
zernike_percentage = df_zernike_50pct.groupby(species_column).apply(lambda x: (x.iloc[:, 3:].sum() / len(x)) * 100)

# Group by species and calculate the percentage of 1s for each column in Pseudo-Zernike
pseudozernike_percentage = df_pseudozernike_50pct.groupby(species_column).apply(lambda x: (x.iloc[:, 3:].sum() / len(x)) * 100)

# Combine the Experimental, Zernike, and Pseudo-Zernike matrices into a single probability matrix
probability_matrix = pd.concat([experimental_percentage, zernike_percentage, pseudozernike_percentage], 
                               axis=1, keys=["Experimental", "Zernike", "Pseudo-Zernike"])

# Save the combined probability matrix to a new Excel file
output_file = "results/Probability_Matrix_50pct_Experimental_Zernike_PseudoZernike.xlsx"
probability_matrix.to_excel(output_file, engine='openpyxl', index=True)

# Display the first few rows of the matrix for verification
print("Probability Matrix (Percentage Positives for Experimental, Zernike, and Pseudo-Zernike at 50% Threshold):")
print(probability_matrix.head())

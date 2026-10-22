import pandas as pd
import numpy as np
from scipy.stats import norm  # For Gaussian Integral calculation

# Define file paths for input and output
input_path = "results/Probability_Matrix_50pct_Experimental_Zernike_PseudoZernike.xlsx"
output_path = "results/Debugged_Results.xlsx"

# Load the probability matrix, skipping empty rows and handling headers manually
prob_matrix = pd.read_excel(input_path, skiprows=[0, 2])  # Skip row 1 and row 3
prob_matrix.rename(columns={prob_matrix.columns[0]: "Taxon"}, inplace=True)  # Rename the first column to 'Taxon'
prob_matrix.set_index("Taxon", inplace=True)  # Set 'Taxon' column as the index

# Define the number of strains for each taxon
num_strains = {
    "A. butzleri cryaerophilus Type 2": 3,
    "A.aquimarinus": 1,
    "A.butzleri Type 1": 3,
    "A.cibarius": 1,
    "A.cloacae": 1,
    "A.cryaerophilus Type 1": 5,
    "A.defluvii": 1,
    "A.mytili": 1,
    "A.skirrowii": 1,
    "A.venerupis": 1,
    "Aeromonas Pork 22 hours 2x": 2,
    "Ecoli 12h": 10,
    "y.kristensenii 22h 2x": 3,
    "yersinia enterocolitica 22h 2x": 7,
    "yersinia frederiksenii 22h 2x": 1,
    "yersinia intermedia 22h 2x": 3,
    "yersinia pseudotuberclosis 22h 2x": 4,
}

# Normalize the probability matrix
normalized_matrix = (prob_matrix - prob_matrix.min()) / (prob_matrix.max() - prob_matrix.min())

# Function to create the HMO for each taxon
def calculate_hmo(taxon_data, threshold=0.5):
    return (taxon_data >= threshold).astype(int)

# Function to calculate Willcox Probability
def calculate_willcox_prob(hmo, prob_matrix, num_strains):
    willcox_probs = {}
    for j_taxon in prob_matrix.index:
        n = num_strains.get(j_taxon, 1)
        b_j = n / (n - 1) if n > 1 else 1  # Correction Factor
        likelihood = np.prod(np.abs(hmo + prob_matrix.loc[j_taxon] - 1)) * b_j
        willcox_probs[j_taxon] = likelihood
    total_likelihood = sum(willcox_probs.values())
    willcox_probs = {taxon: likelihood / total_likelihood for taxon, likelihood in willcox_probs.items()}  # Normalize
    return willcox_probs

# Function to calculate Taxonomic Distance and Gaussian Integral
def calculate_taxonomic_distance_and_gaussian(hmo, prob_matrix, num_strains):
    distances = {}
    gaussian_integrals = {}
    for j_taxon in prob_matrix.index:
        n = num_strains.get(j_taxon, 1)
        p_j = prob_matrix.loc[j_taxon].mean()  # Mean of character states for taxon
        m = len(prob_matrix.columns)  # Number of features

        # Distance calculation
        distance = np.sqrt(np.sum((hmo - prob_matrix.loc[j_taxon]) ** 2) / m)
        distances[j_taxon] = distance

        # Gaussian Integral calculation
        d2_j = 2 * p_j * (1 - p_j) / m  # Mean squared distance
        if n > 1:
            d2_j *= n / (n - 1)  # Correct for degrees of freedom
        d_j = np.sqrt((2 * m - 1) * d2_j / (2 * m))  # Approximation of mean distance
        s_j = np.sqrt(d2_j / (2 * m))  # Standard deviation
        se_d = (distance - d_j) / s_j if s_j > 0 else 0  # Standard Error
        gaussian_integrals[j_taxon] = norm.cdf(se_d)  # Gaussian Integral
    return distances, gaussian_integrals

# Test thresholds of 0.6, 0.7, and 0.4
thresholds_to_test = [0.6, 0.7, 0.4]
debug_results = []

for threshold in thresholds_to_test:
    print(f"\nTesting Threshold: {threshold}")
    for taxon in prob_matrix.index:
        hmo = calculate_hmo(normalized_matrix.loc[taxon], threshold)
        willcox_probs = calculate_willcox_prob(hmo, normalized_matrix, num_strains)
        distances, gaussian_integrals = calculate_taxonomic_distance_and_gaussian(hmo, normalized_matrix, num_strains)
        
        # Get top 3 best matches based on Willcox probabilities
        top_3_matches = sorted(willcox_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top_matches_str = ", ".join([f"{match[0]} ({match[1]:.4f})" for match in top_3_matches])
        
        # Determine closest taxon based on minimum distance
        closest_taxon = min(distances, key=distances.get)
        min_distance = distances[closest_taxon]
        
        # Calculate minimum distances for the top 3 matches
        min_distances_top_3 = {taxon: distances[taxon] for taxon, _ in top_3_matches}

        debug_results.append({
            "Taxon": taxon,
            "Threshold": threshold,
            "Best Match (Willcox)": top_3_matches[0][0],
            "Closest Taxon (Distance)": closest_taxon,
            "Minimum Distance": min_distance,
            "Gaussian Integral": gaussian_integrals[closest_taxon],
            "Top 3 Matches (Willcox Probabilities)": top_matches_str,
            "Top 3 Minimum Distances": min_distances_top_3,
            "Agreement": top_3_matches[0][0] == closest_taxon
        })

# Create a DataFrame for debugging results
debug_results_df = pd.DataFrame(debug_results)

# Save results
debug_results_df.to_excel(output_path, index=False)

# Print summary
print("\nDebug Results Summary:")
print(debug_results_df.groupby('Threshold')['Agreement'].mean())

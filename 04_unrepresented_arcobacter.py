import pandas as pd
import numpy as np

# Load identification matrix and binary test data
matrix_file = 'results/Probability_Matrix_50pct_Experimental_Zernike_PseudoZernike.xlsx'
test_data_files = {
    'Experimental': 'data/Test_Data_Experimental_Binary.xlsx',
    'Pseudo-Zernike': 'data/Test_Data_PseudoZernike_Binary.xlsx',
    'Zernike': 'data/Test_Data_Zernike_Binary.xlsx'
}
output_file = 'results/Wilcox_Probabilities_Combined.xlsx'

def load_probability_matrix(file_path):
    matrix = pd.read_excel(file_path, skiprows=1)
    matrix.rename(columns={matrix.columns[0]: 'Taxon'}, inplace=True)
    matrix = matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
    matrix.set_index('Taxon', inplace=True)
    matrix = matrix.apply(pd.to_numeric, errors='coerce')
    epsilon = 1e-2
    return matrix.replace(0, epsilon).replace(1, 1 - epsilon)

identification_matrix = load_probability_matrix(matrix_file)

# Load and combine test data
combined_test_data = None
for feature_type, test_file in test_data_files.items():
    df = pd.read_excel(test_file).dropna(axis=1, how='all').fillna(0)
    cols = ['Taxon'] + list(identification_matrix.columns)
    df = df.loc[:, df.columns.intersection(cols)]
    combined_test_data = df if combined_test_data is None else pd.merge(combined_test_data, df, on='Taxon', how='outer')

combined_test_data = combined_test_data[['Taxon'] + list(identification_matrix.columns)]

def calculate_wilcox_probabilities(matrix, test_cases):
    results = []
    matrix_probs = matrix.values
    taxon_names = matrix.index.values

    for _, test_case in test_cases.iterrows():
        tc_id = str(test_case['Taxon'])
        binary_data = test_case.drop(labels=['Taxon'], errors='ignore').fillna(0).astype(float).values

        # compute log-likelihoods for each reference taxon
        log_likelihoods = []
        for group_probs in matrix_probs:
            with np.errstate(divide='ignore', invalid='ignore'):
                lp = np.where(binary_data == 1,
                              np.log(group_probs),
                              np.log(1 - group_probs))
                lp = np.nan_to_num(lp, nan=0.0, neginf=0.0, posinf=0.0)
            log_likelihoods.append(lp.sum())

        # convert to normalized probabilities via log-sum-exp
        log_likelihoods = np.array(log_likelihoods)
        max_log = log_likelihoods.max()
        exp_ls = np.exp(log_likelihoods - max_log)
        probs = exp_ls / exp_ls.sum() if exp_ls.sum() > 0 else np.zeros_like(exp_ls)

        # find predicted taxon and its log-likelihood
        pred_idx = log_likelihoods.argmax()
        pred_taxon = taxon_names[pred_idx]
        pred_logL = log_likelihoods[pred_idx]

        # record: Test Case ID, Predicted Taxon, Predicted Log-Likelihood, then all probabilities
        results.append([tc_id, pred_taxon, pred_logL] + list(np.round(probs, 4)))

    cols = ['Test Case ID', 'Predicted Taxon', 'Predicted Log-Likelihood'] + list(taxon_names)
    return pd.DataFrame(results, columns=cols)

# run and save
wilcox_results = calculate_wilcox_probabilities(identification_matrix, combined_test_data)
wilcox_results.to_excel(output_file, index=False)
print(f"Saved results (with log-likelihood) to {output_file}")

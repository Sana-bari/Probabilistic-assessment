import pandas as pd
import numpy as np

# === File paths (update if necessary) ===
EXP_BIN_FILE      = 'results/50%_Experimental_Binary_Conversion.xlsx'
ZERNIKE_BIN_FILE  = 'results/50%_Zernike_Moments_Binary_Conversion.xlsx'
PSEUDO_BIN_FILE   = 'results/50%_Pseudo-Zernike_Moments_Binary_Conversion.xlsx'

PROB_MATRIX_FILE  = 'results/Probability_Matrix_50pct_Experimental_Zernike_PseudoZernike_Smoothed.xlsx'
RESULTS_FILE      = 'results/Colony_LOO_All_Species.xlsx'

# === Species for leave-one-out (one random colony each) ===
species_list = [
    'butzleri cryaerophilus Type 2',
    'aquimarinus',
    'butzleri Type 1',
    'cloacae',
    'cryaerophilus Type 1',
    'mytili',
    'skirrowii',
    'venerupis',
    'defluvii',
    'enterocolitica 22h 2x',
    'intermedia 22h 2x'
]

# === Load full binary datasets ===
df_exp    = pd.read_excel(EXP_BIN_FILE)
df_zern   = pd.read_excel(ZERNIKE_BIN_FILE)
df_pseudo = pd.read_excel(PSEUDO_BIN_FILE)

all_records = []

def split(df, test_keys):
    train = (
        df.merge(test_keys, on=['Taxon','Strain ID','Colony ID'], how='left', indicator=True)
          .query('_merge=="left_only"')
          .drop(columns=['_merge'])
    )
    test  = df.merge(test_keys, on=['Taxon','Strain ID','Colony ID'], how='inner')
    return train, test

for sp in species_list:
    # 1) pick one random colony for this species (from experimental table)
    mask = df_exp['Taxon'].str.contains(sp, case=False, na=False)
    candidates = (
        df_exp.loc[mask, ['Taxon','Strain ID','Colony ID']]
              .drop_duplicates()
    )
    if candidates.empty:
        raise ValueError(f"No records found for species containing '{sp}'")
    test_keys = candidates.sample(n=1)   # random choice
    taxon, strain, col_id = (
        test_keys.iloc[0]['Taxon'],
        test_keys.iloc[0]['Strain ID'],
        test_keys.iloc[0]['Colony ID']
    )

    # 2) split each feature‐set into training vs this one test row
    df_exp_tr,    df_exp_te    = split(df_exp,    test_keys)
    df_zern_tr,   df_zern_te   = split(df_zern,   test_keys)
    df_pseudo_tr, df_pseudo_te = split(df_pseudo, test_keys)

    # combine test‐row across all three feature‐sets
    df_test = (
        df_exp_te
        .merge(df_zern_te,   on=['Taxon','Strain ID','Colony ID'])
        .merge(df_pseudo_te, on=['Taxon','Strain ID','Colony ID'])
    )

    # 3) build & smooth probability matrix from training data
    id_cols      = ['Taxon','Strain ID','Colony ID']
    exp_feats    = [c for c in df_exp_tr.columns    if c not in id_cols]
    zern_feats   = [c for c in df_zern_tr.columns   if c not in id_cols]
    pseudo_feats = [c for c in df_pseudo_tr.columns if c not in id_cols]

    exp_pct    = df_exp_tr.groupby('Taxon')[exp_feats].mean()     * 100
    zern_pct   = df_zern_tr.groupby('Taxon')[zern_feats].mean()   * 100
    pseudo_pct = df_pseudo_tr.groupby('Taxon')[pseudo_feats].mean()* 100

    prob_mat = pd.concat([exp_pct, zern_pct, pseudo_pct], axis=1)
    prob_mat = (prob_mat / 100.0).clip(0.01, 0.99)  # Laplace smoothing
    prob_mat.to_excel(PROB_MATRIX_FILE, index=True, engine='openpyxl')

    # 4) compute log‐likelihoods, normalized probs & Euclidean distances
    taxa     = prob_mat.index.tolist()
    features = prob_mat.columns.tolist()
    mat_vals = prob_mat.values

    for _, row in df_test.iterrows():
        binary = row[features].fillna(0).astype(float).values

        # a) log‐likelihood
        logl = []
        for p in mat_vals:
            with np.errstate(divide='ignore', invalid='ignore'):
                lp = np.where(binary==1, np.log(p), np.log(1-p))
                lp = np.nan_to_num(lp, nan=0.0, posinf=0.0, neginf=0.0)
            logl.append(lp.sum())
        logl = np.array(logl)

        # b) normalized probabilities
        exps   = np.exp(logl - logl.max())
        wprobs = exps / exps.sum()

        # c) Euclidean distances
        dists = np.sqrt(((binary - mat_vals)**2).sum(axis=1) / binary.size)

        case_id = f"{strain} | {taxon} | Colony {col_id}"
        for taxon_i, ll_val, prob, dist in zip(taxa, logl, wprobs, dists):
            all_records.append([
                case_id, taxon_i,
                round(ll_val,  4),
                round(prob,    4),
                round(dist,    4)
            ])

# 5) consolidate & save results—**keep** the Test Case column in each sheet
results_df = pd.DataFrame(
    all_records,
    columns=['Test Case','Taxon','LogLikelihood','NormalizedProb','Distance']
)

with pd.ExcelWriter(RESULTS_FILE, engine='openpyxl') as writer:
    for case_id, group in results_df.groupby('Test Case'):
        sheet = case_id.split()[-1]  # Colony ID
        # write all columns, so the very first column tells you exactly which species/strain/colony
        group.to_excel(
            writer,
            sheet_name=f"Colony_{sheet}",
            index=False
        )

print(f"✅ Saved LOO results for {len(species_list)} species to:\n   {RESULTS_FILE}")

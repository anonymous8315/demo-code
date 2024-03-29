import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import pickle
from tqdm import tqdm
import json

from compute_elo import create_matchup_df, get_replicate_ratings, consolidate_replicate_ratings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['random', 'protein', 'species', 'both'], default='random')
    parser.add_argument('--test_frac', type=float, default=0.3)
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument('--model', choices=['rf', 'knn'], default='knn')
    parser.add_argument('--metric', choices=['progen2-xlarge_fp16_False_ll', 'ESM2_15B_pppl'], default='ESM2_15B_pppl')
    args = parser.parse_args()

    verbose=False

    # ===================================================================================
    # ====================      Prepare data and load embeddings      ===================
    # ===================================================================================

    # load data
    df = pd.read_csv("human_bias_data/common_euk_gene_seqs_w_progen_esm_ll.csv")
    unique_df = df.drop_duplicates(subset=['genus_species', 'sequence']).reset_index()
    unique_idxs = df.drop_duplicates(subset=['genus_species', 'sequence']).index.values

    model = 'progen2-xlarge'
    seqs = 'common_euk_gene_seqs_w_progen_esm_ll'
    with open(f'human_bias_data/{model}_{seqs}_embeddings_avg_positions_True.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    embeddings = np.array(embeddings)
    unique_embeddings = embeddings[unique_idxs, :, :]

    # set ground truth labels for correction
    metric = args.metric
    correction_column = 'correction'
    df[correction_column] = 0.0
    for protein in df.first_protein_name.unique():
        max_ll = df[df.first_protein_name == protein][metric].max()
        df.loc[df.first_protein_name == protein, correction_column] = max_ll
    df[correction_column] = df[correction_column] - df[metric]

    y = df[correction_column][unique_idxs].values

    # setup for random splits
    protein_names = unique_df.first_protein_name.unique()
    species_names = unique_df.genus_species.unique()
    train_frac = 1 - args.test_frac

    results_dict = defaultdict(list)

    # =========================================================================================
    # ================      Run data splitting, training, and elo computation      ============
    # =========================================================================================

    for run_idx in tqdm(range(args.num_trials)):
        rng = np.random.default_rng(args.random_state + 123*run_idx)
        # define train and test distribution
        if args.split == 'random':
            num_train = int(train_frac * len(unique_df))
            permuted_idxs = rng.permutation(list(range(len(unique_df))))
            idxs_train = permuted_idxs[:num_train]
            idxs_test = permuted_idxs[num_train:]
            print('Num train rows:', len(idxs_train), 'Num test rows:', len(idxs_test))
            # X_train, X_test, y_train, y_test = train_test_split(unique_embeddings[:, args.layer, :], y, test_size=train_frac, random_state=args.random_state + 1000*run_idx)

        elif args.split == 'protein':
            num_train = int(train_frac * len(protein_names))
            permuted_proteins =  rng.permutation(protein_names)
            train_proteins = permuted_proteins[:num_train]
            test_proteins = permuted_proteins[num_train:]

            idxs_train = unique_df[unique_df.first_protein_name.isin(train_proteins)].index.values
            idxs_test = unique_df[unique_df.first_protein_name.isin(test_proteins)].index.values
            print('Num train rows:', len(idxs_train), 'Num test rows:', len(idxs_test))

        elif args.split == 'species':
            num_train = int(train_frac * len(species_names))
            permuted_species =  rng.permutation(species_names)
            train_species = permuted_species[:num_train]
            test_species = permuted_species[num_train:]

            idxs_train = unique_df[unique_df.genus_species.isin(train_species)].index.values
            idxs_test = unique_df[unique_df.genus_species.isin(test_species)].index.values
            print('Num train rows:', len(idxs_train), 'Num test rows:', len(idxs_test))

        elif args.split == 'both':
            this_train_frac = 1 - np.sqrt(args.test_frac)
            num_train_p = int(this_train_frac * len(protein_names))
            num_train_s = int(this_train_frac * len(species_names))
            permuted_proteins = rng.permutation(protein_names)
            permuted_species = rng.permutation(species_names)
            train_proteins = permuted_proteins[:num_train_p]
            test_proteins = permuted_proteins[num_train_p:]
            train_species = permuted_species[:num_train_s]
            test_species = permuted_species[num_train_s:]

            idxs_train = unique_df[(unique_df.first_protein_name.isin(train_proteins)) & (unique_df.genus_species.isin(train_species))].index.values
            idxs_test = unique_df[(unique_df.first_protein_name.isin(test_proteins)) &  (unique_df.genus_species.isin(test_species))].index.values
            print('Num train rows:', len(idxs_train), 'Num test rows:', len(idxs_test))

        X_train = unique_embeddings[idxs_train, args.layer, :]
        X_test = unique_embeddings[idxs_test, args.layer, :]
        y_train = y[idxs_train]
        y_test = y[idxs_test]

        # train regressor
        if args.model == 'rf':
            model = RandomForestRegressor(max_features = 'sqrt')
        elif args.model == 'knn':
            model = KNeighborsRegressor(n_neighbors=5, weights='uniform')
        model.fit(X_train, y_train)
        print('Train score: ', model.score(X_train, y_train), 'Test score: ', model.score(X_test, y_test))

        corrected_df = pd.DataFrame(unique_df.iloc[idxs_test])
        corrected_df['ll_correction'] = model.predict(X_test) 
        corrected_df['corrected_ll'] = corrected_df['ll_correction'] + corrected_df[args.metric]

        # compute Elo ratings
        metrics=['corrected_ll', args.metric]
        matchup_df = create_matchup_df(corrected_df, metrics=metrics)

        # make multiple permutations of the matchup df to compute Elo
        ratings_replicates = get_replicate_ratings(matchup_df, num_replicates=10, metrics=metrics)
        ratings_df = consolidate_replicate_ratings(ratings_replicates, metrics=metrics)
    
        # consolidate to a set of mean Elo ratings
        mean_ratings_df = ratings_df.groupby(by='genus_species', as_index=False).agg(['mean'])
        mean_ratings_df.columns = mean_ratings_df.columns.get_level_values(0)
        if verbose:
            mean_ratings_df.to_csv(f'test_elo_{run_idx}.csv', index=False)
        print(mean_ratings_df.describe())

        # compile results
        results_dict['corrected_std'].append( mean_ratings_df['Elo_corrected_ll'].std() )
        results_dict['corrected_iqr'].append( mean_ratings_df['Elo_corrected_ll'].quantile(.75) -  mean_ratings_df['Elo_corrected_ll'].quantile(.25))
        results_dict['og_std'].append( mean_ratings_df[f'Elo_{args.metric}'].std() )
        results_dict['og_iqr'].append( mean_ratings_df[f'Elo_{args.metric}'].quantile(.75) -  mean_ratings_df[f'Elo_{args.metric}'].quantile(.25))

    # ==================================================================================
    # ================      Compile results, save for further analysis      ============
    # ==================================================================================
    
    out_json = "correction_results/elo_correction_results_" + "_".join([f"{key}_{val}" for key, val in vars(args).items()]) + ".json"
    with open(out_json, 'w') as f:
        json.dump(results_dict, f)

if __name__ == '__main__':
    main()

import csv
import math
import argparse

import numpy as np
import pandas as pd

def create_matchup_df(df, metrics):
    rows = []
    for protein in df.first_protein_name.unique():
        sub_df = df[df.first_protein_name == protein]
        for idx, species_1 in enumerate(sub_df.genus_species.values):
            for species_2 in sub_df.genus_species.values[idx+1:]:
                row = {'first_protein_name': protein, 'species_1': species_1, 'species_2': species_2}
                for metric in metrics:
                    score_1 = np.median(list(sub_df[sub_df.genus_species == species_1][metric].values))
                    score_2 = np.median(list(sub_df[sub_df.genus_species == species_2][metric].values))
                    if score_1 > score_2:
                        result = 1
                    elif score_1 < score_2:
                        result = 0
                    else:
                        result = 0.5
                    row[metric] = result
                rows.append(row)

    matchup_df = pd.DataFrame(rows)
    return matchup_df

# Function to compute the expected outcome of a game between two players
def expected_outcome(rating1, rating2):
    return 1 / (1 + math.pow(10, (rating2 - rating1) / 400))

# Function to update the Elo rating for a player after a game
def update_elo(player_rating, opponent_rating, result, k=32):
    expected = expected_outcome(player_rating, opponent_rating)
    new_rating = player_rating + k * (result - expected)
    return new_rating

# Function to compute Elo ratings from a pandas DataFrame and return the updated ratings
def compute_elo_from_dataframe(df, initial_ratings=None, k=32, metrics=['progen2-xlarge_fp16_False_ll']):
    player_ratings = {}
    for metric in metrics:
        player_ratings[metric] = {}
        if initial_ratings is not None:
            player_ratings[metric].update(initial_ratings)
    
    
    for index, row in df.iterrows():
        player1 = row['species_1']
        player2 = row['species_2']
        for metric in metrics:
            result = row[metric]

            if player1 not in player_ratings[metric]:
                player_ratings[metric][player1] = 1500  # Initial rating if not provided
            if player2 not in player_ratings[metric]:
                player_ratings[metric][player2] = 1500  # Initial rating if not provided

            player1_rating = player_ratings[metric][player1]
            player2_rating = player_ratings[metric][player2]

            # Update Elo ratings for both players
            new_player1_rating = update_elo(player1_rating, player2_rating, result, k)
            new_player2_rating = update_elo(player2_rating, player1_rating, 1 - result, k)

            player_ratings[metric][player1] = new_player1_rating
            player_ratings[metric][player2] = new_player2_rating
    return player_ratings

def get_replicate_ratings(matchup_df, metrics = ['progen2-medium_fp16_False_ll',
                                                'progen2-base_fp16_False_ll',
                                                'progen2-large_fp16_False_ll',
                                                'progen2-xlarge_fp16_False_ll',
                                                'progen2-BFD90_fp16_False_ll',
                                                'ESM2_650M_pppl',
                                                'ESM2_3B_pppl',
                                                'ESM2_15B_pppl'
                                                ], 
                                                num_replicates = 50, k=16):

    ratings_replicates = []
    for idx in range(num_replicates):
        this_ratings = compute_elo_from_dataframe(matchup_df.sample(frac=1, random_state=idx),
                                                 k=k, metrics=metrics)
        ratings_replicates.append(this_ratings)
    return ratings_replicates

def consolidate_replicate_ratings(ratings_replicates, metrics):
    ratings_df_replicates = []
    for this_ratings in ratings_replicates:
        ratings_df = []
        for metric in metrics:
            this_ratings_df = pd.DataFrame.from_dict(this_ratings[metric], orient='index', columns=[f'Elo_{metric}'])
            this_ratings_df.reset_index(inplace=True)
            this_ratings_df.rename(columns={'index': 'genus_species'}, inplace=True)
            if len(ratings_df) == 0:
                # start it off
                ratings_df = this_ratings_df
            else:
                # merge
                ratings_df = ratings_df.merge(this_ratings_df, on='genus_species')
        ratings_df_replicates.append(ratings_df)
    return pd.concat(ratings_df_replicates)

def main():

    # params

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_ll_csv', type=str)
    parser.add_argument('--metric', type=str)
    parser.add_argument('--out_replicate_csv', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.seq_ll_csv)

    # create matchup df
    matchup_df = create_matchup_df(df, metrics=[args.metric])

    # compute Elo ratings with multiple permutations of the matchup df
    ratings_replicates = get_replicate_ratings(matchup_df, metrics=[args.metric])
    ratings_df = consolidate_replicate_ratings(ratings_replicates, metrics=[args.metric])
    # ratings_df.to_csv(args.out_replicate_csv, index=False)

   
    # consolidate to a set of mean Elo ratings
    mean_ratings_df = ratings_df.groupby(by='genus_species', as_index=False).agg(['mean'])
    mean_ratings_df.columns = mean_ratings_df.columns.get_level_values(0)

    return mean_ratings_df

# Built on top of https://github.com/salesforce/progen/blob/main/progen2/likelihood.py
# See BSD 3-Clause License here: https://github.com/salesforce/progen/blob/main/LICENSE.txt

import os
import copy
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import pickle
import torch
import argparse
from likelihood import log_likelihood, print_time, set_env, set_seed, create_model, create_tokenizer_custom

# Constants
aa_vocab = [
    "A",
    #"B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    #"O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    #"U",
    "V",
    "W",
    #"X",
    "Y",
    #"Z",
]

def main():
    # (0) constants

    models_151M = [ 'progen2-small' ]
    models_754M = [ 'progen2-medium', 'progen2-oas', 'progen2-base' ]
    models_2B = [ 'progen2-large', 'progen2-BFD90' ]
    models_6B = [ 'progen2-xlarge' ]
    models = models_151M + models_754M + models_2B + models_6B


    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, default='progen2-base')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, default='1MGHGVSRPPVVTLRPAVLDDCPVLWRWRNDPETRQASVDEREIPVDTHTRWFEETLKRFDRKLFIVSADGVDAGMVRLDIQDRDAAVSVNIAPEWRGRGVGPRALGCLSREAFGPLALLRMSAVVKRENAASRIAFERAGFTVVDTGGPLLHSSKARLHVVAAIQARMGSTRLPGKVLVSIAGRPTIQRIAERLAVCQELDAVAVSTSVENRDDAIADLAAHLGLVCVRGSETDLIERLGRTAARTGADALVRITADCPLVDPALVDRVVGVWRRSAGRLEYVSNVFPPTFPDGLDVEVLSRTVLERLDREVSDPFFRESLTAYVREHPAAFEIANVEHPEDLSRLRWTMDYPEDLAFVEAVYRRLGNQGEIFGMDDLLRLLEWSPELRDLNRCREDVTVERGIRGTGYHAALRARGQAP2')
    parser.add_argument('--sanity', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--lam', type=float, default=1, help='lambda scaling fitness vs. entropy')
    parser.add_argument('--lam_multiplier', type=float, default=1.1, help='factor to increase lambda at intervals')
    parser.add_argument('--lam_multiplier_freq', type=int, default=400, help='frequency to increase lambda')
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--sample_freq', type=int, default=100)
    parser.add_argument('--outfile', type=str, default='seq_trajectory')
    parser.add_argument('--verbose', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--correction_model_fname')
    args = parser.parse_args()


    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    ckpt = f'./checkpoints/{args.model}'

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False


    # (3) load

    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=args.fp16).to(device)


    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')




    def ll(tokens, f=log_likelihood, reduction='mean', return_embedding=False):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.fp16):
                target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
                if return_embedding:
                    outputs = model(target, labels=target, output_hidden_states=True)
                else:
                    outputs = model(target, labels=target)
                logits = outputs.logits
                if return_embedding:
                    embedding = outputs.hidden_states[-1]
                    embedding = embedding.mean(dim=-2)
                
                # shift
                logits = logits[:-1, ...]
                target = target[1:]

                # remove terminals
                bos_token, eos_token = 3, 4
                if target[-1] in [bos_token, eos_token]:
                    logits = logits[:-1, ...]
                    target = target[:-1]

                assert (target == bos_token).sum() == 0
                assert (target == eos_token).sum() == 0

                # remove unused logits
                first_token, last_token = 5, 29
                logits = logits[:, first_token:(last_token+1)]
                target = target - first_token

                assert logits.shape[1] == (last_token - first_token + 1)

                if return_embedding:
                    return f(logits=logits, target=target, reduction=reduction).item(), embedding

                else:
                    return f(logits=logits, target=target, reduction=reduction).item(), None

    def log_likelihood_sum(tokens, return_embedding=False):
        reverse = lambda s: s[::-1]
        ll_lr_sum, embedding = ll(tokens=tokens, reduction='sum', return_embedding=return_embedding)
        ll_rl_sum, _ = ll(tokens=reverse(tokens), reduction='sum')
        ll_sum = .5 * (ll_lr_sum + ll_rl_sum)
        
        return ll_sum, embedding
    
    def log_likelihood_mean(tokens, return_embedding=False):
        reverse = lambda s: s[::-1]
        ll_lr_mean, embedding = ll(tokens=tokens, reduction='mean', return_embedding=return_embedding)
        ll_rl_mean, _ = ll(tokens=reverse(tokens), reduction='mean')
        ll_mean = .5 * (ll_lr_mean + ll_rl_mean)
        
        return ll_mean, embedding

    def gibbs_step(seq, vocab, lam, correction_model=None, verbose=False):
        # randomly select index
        gen_idx = np.random.randint(1, len(seq)-1)
        
        # calculate vector of probabilities  p(X_i | x_{-i})
        logits_xi = np.zeros(len(vocab))
        for aa_idx, aa in enumerate(vocab):
            mut_seq = copy.deepcopy(list(seq))
            mut_seq[gen_idx] = aa
            mut_seq = ''.join(mut_seq)
            temp_logits, embedding = log_likelihood_sum(tokens=mut_seq, return_embedding=True)
            logits_xi[aa_idx] = lam * temp_logits            
            if correction_model:
                mut_embedding = embedding.detach().cpu().numpy()
                ll_correction = correction_model.predict(mut_embedding.reshape(1, -1))
                logits_xi[aa_idx] += lam * ll_correction.item() * (len(mut_seq)-2)
        prob_xi = softmax(logits_xi)

        # sample from mutations according to probability vector
        new_aa = np.random.choice(a=vocab, p=prob_xi)
        new_seq = copy.deepcopy(list(seq))
        new_seq[gen_idx] = new_aa
        if verbose:
            if correction_model:
                print('latest ll correction: ', ll_correction)
                print('mut_embedding shape: ', mut_embedding.shape)
            print('pre-normalization logits: ', logits_xi)
            print('post-normalization P(X_i | x_{-i}): ', prob_xi)
            print('idx to generate: ', gen_idx)
            print('old AA, new AA: ', seq[gen_idx], new_aa)
            print('=======================')
            print('new seq: ', ''.join(new_seq))

        return ''.join(new_seq)

    def get_embedding(tokens):
        if len(tokens) > 1022:
            tokens = f"1{tokens[:args.max_length+1]}"
        else:
            tokens = f"1{tokens}2"
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.fp16):
                target=torch.tensor(tokenizer.encode(tokens).ids).to(device)
                hidden_states = model(target, output_hidden_states=True).hidden_states
                return hidden_states

    def embedding_to_numpy(embeddings, avg_positions=True):
        out_list = []
        for embedding in embeddings:
            np_embedding = embedding.detach().cpu().numpy()
            if embedding.size(dim=0) == 1:
                np_embedding = np_embedding[0, :, :]
            if avg_positions:
                np_embedding = np_embedding.mean(axis=-2)
            out_list.append(np_embedding)
        return np.array(out_list)

    def predict_ll_correction(correction_model, seq):
        mut_embedding = embedding_to_numpy(get_embedding(tokens = seq), avg_positions=True)[-1,:]
        ll_correction = correction_model.predict(mut_embedding)
        print(ll_correction)
        return ll_correction.item()

    # RUN DESIGN
    if args.correction_model_fname:
        correction_model = pickle.load(open(args.correction_model_fname, 'rb'))
    else:
        correction_model = None

    seq_trajectory = [args.context[1:-1]]
    new_seq = args.context 
    starting_ll, starting_embedding = log_likelihood_mean(args.context, return_embedding=True)
    ll_trajectory = [starting_ll]
    if correction_model:
        starting_embedding = starting_embedding.detach().cpu().numpy()
        corrected_ll_trajectory = [starting_ll + correction_model.predict(starting_embedding.reshape(1, -1)).item()]

    lam = args.lam
    for i in range(args.num_steps):
        new_seq = gibbs_step(seq=new_seq, 
                             vocab=aa_vocab, 
                             lam=lam,
                             correction_model=correction_model,
                             verbose=args.verbose)
        if i % args.sample_freq == 0:
            seq_trajectory.append(new_seq[1:-1])
            this_ll_mean, this_embedding = log_likelihood_mean(new_seq, return_embedding=True)
            ll_trajectory.append(this_ll_mean)
            print(i, new_seq[1:-1], flush=True)
            print('ll_mean:', ll_trajectory[-1], flush=True)
            if correction_model:
                this_embedding = this_embedding.detach().cpu().numpy()
                corrected_ll_trajectory.append(this_ll_mean + correction_model.predict(this_embedding.reshape(1, -1)).item())
                print('corrected_ll_mean:', corrected_ll_trajectory[-1], flush=True)
        if i % args.lam_multiplier_freq == 0:
            lam = lam * args.lam_multiplier
    
    seq_df = pd.DataFrame({'sequence': seq_trajectory, 'll_mean': ll_trajectory, 'corrected_ll_mean': corrected_ll_trajectory})
    seq_df.to_csv(f'{args.outfile}.csv') 

if __name__ == '__main__':
    main()
    print('done.')

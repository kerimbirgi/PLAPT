from plapt import Plapt
from tqdm import tqdm
import pandas as pd
import pickle
import argparse

def evaluate(data_path, protein_path):
    eval_df = pd.read_csv(data_path)
    with open(protein_path, 'rb') as f:
        protein_sequences = pickle.load(f)

    plapt = Plapt()

    predictions = []
    targets = []
    ids = []

    loss = 0
    count = 0
    for idx, row in tqdm(eval_df.iterrows(), total = len(eval_df), desc="Processing Samples"):
        protein_id = row['Target ChEMBLID']
        protein_seq = protein_sequences[protein_id]
        ligand_id = row['Molecule ChEMBLID']
        ligand_SMILE = [row['Molecule SMILES']]

        results = plapt.score_candidates(protein_seq, ligand_SMILE)
        result_dict = results[0]
        predictions.append(result_dict['neg_log10_affinity_M'])
        targets.append(row['-logAffi'])
        loss = result_dict['neg_log10_affinity_M'] - row['-logAffi']
        count += 1
        print(loss / count)
        ids.append(row['Target ChEMBLID'] + '_' + row['Molecule ChEMBLID'])

    return predictions, targets, ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/data2/BindingNetv2/processed/splits/subset_20p_with_false/test.csv" ,type=str, required=True)
    parser.add_argument("--protein_path", default="/data2/BindingNetv2/processed/proteins/first_chain_protein_sequences.pkl" ,type=str, required=True)
    parser.add_argument("--output_csv", default="/data2/home/kgb24/bindingnet_research/outputs/predictions/PLAPT_20p_with_false/predictions.csv" ,type=str, required=True, help="Path to save the output CSV file")
    args = parser.parse_args()
    
    predictions, targets, ids = evaluate(args.data_path, args.protein_path)
    
    # Save to CSV in the order: ids, predictions, targets
    df = pd.DataFrame({
        'ids': ids,
        'predictions': predictions,
        'targets': targets
    })
    df.to_csv(args.output_csv, index=False)
    

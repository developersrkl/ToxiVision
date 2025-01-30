# Author: Shaurya K, Rutgers NB

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import numpy as np

def smiles_to_images(smiles, out_path, size=(200, 200)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=size)
    img.save(out_path)
    return True

def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def clean_dataset(dataset, invalid_list, dataset_name): #remove invalid smiles that were in Tox21 dataset
    valid_rows = []
    for _, row in dataset.iterrows():
        smi = row['smiles']
        mol_id = row['mol_id']
        if validate_smiles(smi):
            valid_rows.append(row)
        else:
            invalid_list.append((mol_id, smi))
    cleaned_df = pd.DataFrame(valid_rows)
    print(f"{dataset_name}: removed {len(invalid_list)} invalid SMILES.") #show what was invalid
    return cleaned_df

def main():
    train_set = pd.read_csv('csv_bin/train.csv')
    val_set = pd.read_csv('csv_bin/val.csv')
    test_set = pd.read_csv('csv_bin/test.csv')

    invalid_smiles_train = []
    invalid_smiles_val = []
    invalid_smiles_test = []

    train_set_cleaned = clean_dataset(train_set, invalid_smiles_train, "Train")
    val_set_cleaned = clean_dataset(val_set, invalid_smiles_val, "Val")
    test_set_cleaned = clean_dataset(test_set, invalid_smiles_test, "Test")

    train_set_cleaned.to_csv('csv_bin/cleaned_train.csv', index=False)
    val_set_cleaned.to_csv('csv_bin/cleaned_val.csv', index=False)
    test_set_cleaned.to_csv('csv_bin/cleaned_test.csv', index=False)

    os.makedirs("training_images", exist_ok=True)
    os.makedirs("val_images", exist_ok=True)
    os.makedirs("test_images", exist_ok=True)

    for _, row in train_set_cleaned.iterrows():
        smi = row['smiles']
        mol_id = row['mol_id']
        out_path = os.path.join("training_images", f"{mol_id}.png")
        smiles_to_images(smi, out_path)

    for _, row in val_set_cleaned.iterrows():
        smi = row['smiles']
        mol_id = row['mol_id']
        out_path = os.path.join("val_images", f"{mol_id}.png")
        smiles_to_images(smi, out_path)

    for _, row in test_set_cleaned.iterrows():
        smi = row['smiles']
        mol_id = row['mol_id']
        out_path = os.path.join("test_images", f"{mol_id}.png")
        smiles_to_images(smi, out_path)

    label_cols = [
        'NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER',
        'NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53'
    ]

    # For train
    train_set_cleaned['img_path'] = train_set_cleaned['mol_id'].apply(
        lambda x: f"training_images/{x}.png"
    )
    columns = ['img_path'] + label_cols
    train_set_cleaned[columns].to_csv("csv_bin/train_labels.csv", index=False)

    # For val
    val_set_cleaned['img_path'] = val_set_cleaned['mol_id'].apply(
        lambda x: f"val_images/{x}.png"
    )
    val_set_cleaned[columns].to_csv("csv_bin/val_labels.csv", index=False)

    # For test
    test_set_cleaned['img_path'] = test_set_cleaned['mol_id'].apply(
        lambda x: f"test_images/{x}.png"
    )
    test_set_cleaned[columns].to_csv("csv_bin/test_labels.csv", index=False)

if __name__ == "__main__":
    main()

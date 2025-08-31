# a function for modifying an input string
# from jiwer import cer
import random
import string
import pandas as pd

def apply_ocr_noise(input_string, target_cer=0.05):
    if type(input_string) != str:
        input_string = str(input_string)
    latin_alphabet_charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ    öüäéèà ÜÄÖ'
    n_changes = int(len(input_string) * target_cer)
    mutated = list(input_string)

    for _ in range(n_changes):
        error_type = random.choice(["substitution", "insertion", "deletion"])

        if error_type == "substitution":
            # Choose a random index to substitute a character
            index = random.randrange(len(mutated))
            mutated[index] = random.choice(latin_alphabet_charset)

        elif error_type == "insertion":
            # Choose a random index to insert a character
            index = random.randrange(len(mutated))
            mutated.insert(index, random.choice(latin_alphabet_charset))

        elif error_type == "deletion":
            # Choose a random index to delete a character, if the string is not empty
            if mutated:
                index = random.randrange(len(mutated))
                del mutated[index]

    return ''.join(mutated)

def apply_noise_to_dataframe(df, target_cer=0.05):
    """
    Applies OCR noise to each cell of a given DataFrame.

    :param df: Pandas DataFrame to be modified
    :param target_cer: Target Character Error Rate for the OCR noise
    :return: Modified DataFrame
    """
    modified_df = df.copy()
    cols = ["doc_text", "query", "summary", "translation"]
    use_cols = [c for c in cols if c in df.columns]

    noised = modified_df[use_cols].applymap(
        lambda cell: apply_ocr_noise(cell, target_cer)
    )

    modified_df = modified_df.join(noised.add_suffix("_noised"))
    
    # for col in modified_df.columns:
    #     if col == 'Index':
    #         continue
    #     for row in range(len(modified_df)):
    #         modified_df.at[row, col] = apply_ocr_noise(modified_df.at[row, col], target_cer)
    return modified_df

# sample_dataset = pd.read_csv('sample_dataset.csv')
sample_dataset_de = pd.read_csv('mlsum_de_val.csv')
sample_dataset_fr = pd.read_csv('mlsum_fr_val.csv')

sample_dataset_corrupted_de = apply_noise_to_dataframe(sample_dataset_de)
sample_dataset_corrupted_fr = apply_noise_to_dataframe(sample_dataset_fr)

sample_dataset_corrupted_de.to_csv('mlsum_de_val_random_noise.csv')
sample_dataset_corrupted_fr.to_csv('mlsum_fr_val_random_noise.csv')

print('datasets saved')

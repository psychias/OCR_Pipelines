# a function for modifying an input string
# from jiwer import cer
import random
import string
import pandas as pd

def apply_ocr_noise(input_string, language, target_cer=0.05):
    if type(input_string) != str:
        input_string = str(input_string)

    russian_alphabet_charset = 'абвгдезийклмнопрстуфхцчшщъыьэюя'
    spanish_alphabet_charset = 'abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZ    áéíóúüÁÉÍÓÚÜ'
    turkish_alphabet_charset = 'abcdefghijklmnopqrstuvwyzABCDEFGHIJKLMNOPQRSTUVYWXZ    öüçğşİı'
    latin_alphabet_charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ    öüäéèà ÜÄÖ'
    n_changes = int(len(input_string) * target_cer)
    mutated = list(input_string)

    for _ in range(n_changes):
        error_type = random.choice(["substitution", "insertion", "deletion"])

        if error_type == "substitution":
            # Choose a random index to substitute a character
            index = random.randrange(len(mutated))
            if language in ['de', 'fr', 'it']:
                mutated[index] = random.choice(latin_alphabet_charset)
            elif language == 'ru':
                mutated[index] = random.choice(russian_alphabet_charset)
            elif language == 'en':
                mutated[index] = random.choice(spanish_alphabet_charset)
            elif language == 'tu':
                mutated[index] = random.choice(turkish_alphabet_charset)

        elif error_type == "insertion":
            # Choose a random index to insert a character
            index = random.randrange(len(mutated))
            if language in ['de', 'fr', 'it']:
                mutated.insert(index, random.choice(latin_alphabet_charset))
            elif language == 'ru':
                mutated.insert(index, random.choice(russian_alphabet_charset))
            elif language == 'en':
                mutated.insert(index, random.choice(spanish_alphabet_charset))
            elif language == 'tu':
                mutated.insert(index, random.choice(turkish_alphabet_charset))

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
    cols = ["text", "summary", "queries"]
    use_cols = [c for c in cols if c in df.columns]

    # Apply noise to each column separately, using the language from each row
    noised_data = {}
    for col in use_cols:
        noised_data[col + "_noised"] = modified_df.apply(
            lambda row: apply_ocr_noise(row[col], row["language"], target_cer), axis=1
        )
    
    # Convert to DataFrame and join with original
    noised_df = pd.DataFrame(noised_data)
    modified_df = modified_df.join(noised_df)
    
    # for col in modified_df.columns:
    #     if col == 'Index':
    #         continue
    #     for row in range(len(modified_df)):
    #         modified_df.at[row, col] = apply_ocr_noise(modified_df.at[row, col], target_cer)
    return modified_df

# sample_dataset = pd.read_csv('sample_dataset.csv')
sample_dataset_= pd.read_csv('q_to_summary.csv')

sample_dataset_corrupted_ = apply_noise_to_dataframe(sample_dataset_)

sample_dataset_corrupted_.to_csv('dataset_random_noise.csv')

print('dataset saved')

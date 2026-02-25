"""
@author: sa-tsdj

"""


import glob
import os

import pandas as pd


DATA_DIR = r'Y:\pc-to-Y\hisco\data\250905\PST2_AUG_RETRAINING'


def find_files(include_gpt: bool = True) -> list[str]:
    # Find HISCO sets
    files = glob.glob(os.path.join(DATA_DIR, 'hisco_inferred/Training_data/*'))

    # Find new "fuzzy" English data
    files.extend(
        [f for f in glob.glob(os.path.join(DATA_DIR, 'new_mappings/fuzzy_duplicates/*')) if 'pst2' in f]
    )

    # Find the "old" but recoded data
    files.append(os.path.join(DATA_DIR, 'pst2_old_recoded/Old_PST2_Training_Data_translated_recoded.csv'))

    # Add the GPT titles
    if include_gpt:
        files.extend(glob.glob(os.path.join(DATA_DIR, 'GPT_titles/*')))

    # Add from scheme
    files.append(os.path.join(DATA_DIR, 'new_pst2_scheme/pst2_summary_descriptions_clean.csv'))

    return files


DTYPES = {
    'pst2_1': str,
    'pst2_2': str,
    'pst2_3': str,
    'pst2_4': str,
    'pst2_5': str,
}


def load(files: list[str]) -> pd.DataFrame:
    data = pd.read_csv(files[0], dtype=DTYPES)
    data['src'] = files[0]

    for f in files[1:]:
        _data = pd.read_csv(f, dtype=DTYPES)
        _data['src'] = f
        data = pd.concat([data, _data])

    return data


def write_all():
    files = find_files()

    # Load
    data = pd.concat([
        pd.read_csv(f, dtype=DTYPES) for f in files
    ])
    assert data['pst2_1'].isna().sum() == 0

    # Seems there is 1 weird NaN observation for "occ1"
    assert data['occ1'].isna().sum() == 1
    data = data[data['occ1'].notna()]

    # Write
    fn_out = r'Z:\faellesmappe\tsdj\hisco\data\Training_data_other\pst2.csv'

    if os.path.isfile(fn_out):
        raise FileExistsError(fn_out)

    data.to_csv(fn_out, index=False)

    # Time for the ugliest of all manual key insertions
    keyset = sorted(set(data['pst2_1']))
    keyset.extend(['?', pd.NA])

    key = pd.DataFrame({
        'system_code': keyset,
        'code': range(len(keyset)),
    })
    key.to_csv(r'Z:\faellesmappe\tsdj\hisco\pst2\mixer-pst2\key_manual.csv', index=False)


def skip_gpt():
    files = find_files(include_gpt=False)

    # Load
    data = pd.concat([
        pd.read_csv(f, dtype=DTYPES) for f in files
    ])
    assert data['pst2_1'].isna().sum() == 0
    assert data['occ1'].isna().sum() == 0

    # Write
    fn_out = r'Z:\faellesmappe\tsdj\hisco\data\Training_data_other\pst2_no_gpt.csv'

    if os.path.isfile(fn_out):
        raise FileExistsError(fn_out)

    data.to_csv(fn_out, index=False)

    # Time for the ugliest of all manual key insertions
    keyset = sorted(set(data['pst2_1']))
    keyset.extend(['?', pd.NA])

    key = pd.DataFrame({
        'system_code': keyset,
        'code': range(len(keyset)),
    })
    key.to_csv(r'Z:\faellesmappe\tsdj\hisco\pst2\mixer-pst2-no-gpt\key_manual.csv', index=False)


def main():
    write_all()




if __name__ == '__main__':
    main()

import os

import pandas as pd

from histocc import OccCANINE, EvalEngine


def predict_file(
        wrapper: OccCANINE,
        data: pd.DataFrame,
        language: str | None = None,
        prediction_type: str = 'greedy',
        threshold: float = 0.17,
) -> pd.DataFrame:
    if language is None:
        language = data['lang']

    results = wrapper.predict(
        data['occ1'],
        lang=language,
        prediction_type=prediction_type,
        threshold=threshold,
    )
    results = results.rename(columns={
        f'{wrapper.system}_{i}': f'{wrapper.system}_{i}_pred' for i in range(1, 6)
    })

    # Add info to differentiate between runs
    results['lang'] = language
    results['pred-type'] = prediction_type
    results['threshold'] = threshold

    return results


def predict_folder(
        wrapper: OccCANINE,
        folder: str,
        save_dir: str | None = None,
) -> pd.DataFrame:
    label_files = os.listdir(folder)

    results = []

    for label_file in label_files:
        data = pd.read_csv(
            os.path.join(folder, label_file),
            keep_default_na=False,
        )

        # for prediction_type in ('flat', 'greedy', 'full'):
        for prediction_type in ('flat', 'greedy'):
            for language in (None, 'unk'):
                _results = predict_file(
                    wrapper=wrapper,
                    data=data,
                    language=language,
                    prediction_type=prediction_type,
                )

                # Add eval scores
                eval_engine = EvalEngine(wrapper, data, _results, wrapper.system)
                _results['acc'] = eval_engine.accuracy(return_per_obs=True)
                _results['recall'] = eval_engine.recall(return_per_obs=True)
                _results['precision'] = eval_engine.precision(return_per_obs=True)
                _results['f1'] = eval_engine.f1(return_per_obs=True)

                # Add label info
                _results = pd.concat([_results, data[[*[f'{wrapper.system}_{i}' for i in range(1, 6)], 'RowID']]], axis=1)

                if save_dir is not None:
                    _results.to_csv(os.path.join(
                        save_dir, f'{label_file}-l={language}-pt={prediction_type}.csv',
                    ), index=False)

                _results['file'] = label_file
                # results.append(_results)

    # results = pd.concat(results)

    return results


def main():
    wrapper = OccCANINE(
        name='OccCANINE_s2s_mix',
        batch_size=4096,
        verbose=False,
    )

    root = r'Z:\faellesmappe\tsdj\hisco\data'
    folder = os.path.join(root, 'Validation_data1')

    results = predict_folder(
        wrapper=wrapper,
        folder=folder,
        save_dir=r'Z:\faellesmappe\tsdj\hisco\results\hisco'
    )

    print(results)


if __name__ == '__main__':
    main()

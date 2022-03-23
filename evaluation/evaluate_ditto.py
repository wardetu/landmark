from functools import partial
from tqdm import tqdm
import pandas as pd
import os
import sys


def compute_explanations():
    prefix = ''
    excluded_cols = ['id','left_id','right_id'] # ['']
    if os.path.expanduser('~') == '/home/baraldian':  # UNI env
        prefix = '/home/baraldian'
    softlab_path = os.path.join(prefix + '/content/drive/Shareddrives/SoftLab/')
    github_code_path = os.path.join(softlab_path, 'Projects/Landmark Explanation EM/Landmark_github')
    code_path = os.path.join(softlab_path, 'Projects/Landmark Explanation EM/Landmark code')
    landmark_path = os.path.join(softlab_path, 'Projects/Landmark Explanation EM')
    dataset_path = os.path.join(softlab_path, 'Dataset', 'Entity Matching')
    base_files_path = os.path.join(softlab_path, 'Projects/Landmark Explanation EM/dataset_files')

    sys.path.append(os.path.join(softlab_path, 'Projects/external_github/ditto'))
    sys.path.append(os.path.join(softlab_path, 'Projects/external_github'))
    sys.path.append(code_path)
    sys.path.append(github_code_path)

    from wrapper.DITTOWrapper import DITTOWrapper
    from landmark import Landmark

    sorted_dataset_names = [
        'BeerAdvo-RateBeer',
        'fodors-zagats',
        'iTunes-Amazon',
        'dirty_itunes_amazon',
        'DBLP-Scholar',
        'dirty_dblp_scholar',
        'walmart-amazon',
        'dirty_walmart_amazon',
        'DBLP-ACM',
        'dirty_dblp_acm',
        'Amazon-Google',
        'Abt-Buy',
    ]
    tasks = [
        'Structured/Beer',
        'Structured/Fodors-Zagats',
        'Structured/iTunes-Amazon',
        'Dirty/iTunes-Amazon',
        'Structured/DBLP-GoogleScholar',
        'Dirty/DBLP-GoogleScholar',
        'Structured/Walmart-Amazon',
        'Dirty/Walmart-Amazon',
        'Structured/DBLP-ACM',
        'Dirty/DBLP-ACM',
        'Structured/Amazon-Google',
        'Textual/Abt-Buy', ]

    checkpoint_path = os.path.join(
        os.path.join(softlab_path, 'Projects/external_github/ditto/checkpoints'))  # 'checkpoints'

    batch_size = 2050
    num_explanations = 100  # 100
    num_samples = 2048  # 2048
    for i in tqdm(range(len(sorted_dataset_names))):

        task = tasks[i]
        turn_dataset_name = sorted_dataset_names[i]
        print('v' * 100)
        print(f'\n\n\n{task: >50}\n' +f'{turn_dataset_name: >50}\n\n\n')
        print('^' * 100)
        turn_dataset_path = os.path.join(dataset_path, turn_dataset_name)
        turn_files_path = os.path.join(base_files_path, turn_dataset_name)
        try:
            os.mkdir(turn_files_path)
        except:
            pass

        dataset_dict = {name: pd.read_csv(os.path.join(turn_dataset_path, f'{name}_merged.csv')) for name in
                        ['train', 'valid', 'test']}

        model = DITTOWrapper(task, checkpoint_path)
        test_df = dataset_dict['test']
        explainer = Landmark(partial(model.predict, batch_size=batch_size), test_df,
                             exclude_attrs=excluded_cols + ['label', 'id'], lprefix='left_', rprefix='right_',
                             split_expression=r' ')
        turn_df = test_df
        pos_mask = turn_df['label'] == 1
        pos_df = turn_df[pos_mask]
        neg_df = turn_df[~pos_mask]
        pos_sample = pos_df.sample(num_explanations, random_state=0) if pos_df.shape[0] >= num_explanations else pos_df
        neg_sample = neg_df.sample(num_explanations, random_state=0) if neg_df.shape[0] >= num_explanations else neg_df

        for conf in ['single', 'double']:
            for sample, prefix in zip([pos_sample, neg_sample], ['positive', 'negative']):
                tmp_path = os.path.join(turn_files_path, f'{prefix}_explanations_{conf}.csv')
                print(f'{prefix} explanations')
                try:
                    # assert False
                    tmp_df = pd.read_csv(tmp_path)
                    assert tmp_df.id.nunique() >= sample.shape[0], 'Not computed'
                    assert False
                    print('loaded')
                except Exception as e:
                    print(e)
                    tmp_df = explainer.explain(sample, num_samples=num_samples, conf=conf)
                    tmp_df.to_csv(tmp_path, index=False)


if __name__ == "__main__":
    compute_explanations()

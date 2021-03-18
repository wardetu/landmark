import numpy as np
import pandas as pd
import ast
from scipy.stats import entropy


def explanation_conversion(explanation_df, item, explainer):
    view = explanation_df[['column', 'position', 'word', 'impact']].reset_index(drop=True)
    tokens_divided = explainer.compute_tokens(item)
    exchanged_idx = [False] * len(view)
    lengths = {col: len(words) for col, words in tokens_divided['tokens'].items()}
    for col, words in tokens_divided['tokens_not_overlapped'].items():  # words injected in the opposite side
        prefix, col_name = col.split('_')
        prefix = 'left_' if prefix == 'right' else 'right_'
        opposite_col = prefix + col_name
        exchanged_idx = exchanged_idx | ((view.position >= lengths[opposite_col]) & (view.column == opposite_col))
    exchanged = view[exchanged_idx]
    view = view[~exchanged_idx]
    # determine injected impacts
    exchanged['side'] = exchanged['column'].apply(lambda x: x.split('_')[0])
    col_names = exchanged['column'].apply(lambda x: x.split('_')[1])
    exchanged['column'] = np.where(exchanged['side'] == 'left', 'right_', 'left_') + col_names
    tmp = view.merge(exchanged, on=['word', 'column'], how='left', suffixes=('', '_injected'))
    tmp = tmp.drop_duplicates(['column', 'word', 'position'], keep='first')
    impacts_injected = tmp['impact_injected']
    impacts_injected = impacts_injected.fillna(0)

    view['score_right_landmark'] = np.where(view['column'].str.startswith('left'), view['impact'], impacts_injected)
    view['score_left_landmark'] = np.where(view['column'].str.startswith('right'), view['impact'], impacts_injected)
    view.drop('impact', 1, inplace=True)

    return view


def compute_metrics(impacts):
  res_dict = {}
  res_dict.update(max= max(impacts),
    min = min(impacts),
    average = np.mean(impacts),
    #entropy = entropy(impacts),
    )
  return res_dict


# convert landmark to json
def explanation_bundle_to_json(item, explanation_df, explainer,
                               exclude_attrs=['id', 'left_id', 'right_id', 'label', 'pred']):
    res_dict = {}
    item_to_display = item.drop(['id', 'left_id', 'right_id'], 1).rename(columns={'pred': 'prediction'}).round(4)
    cols = item_to_display.columns
    item_dict = ast.literal_eval(item_to_display.to_json(orient='records'))[0]
    res_dict.update(
        record_left=' | '.join(item[[col for col in cols if col.startswith('left_')]].astype(str).values[0]),
        record_right=' | '.join(item[[col for col in cols if col.startswith('right_')]].astype(str).values[0]),
        record=item_dict)
    res_dict.update(label=int(item['label'].values[0]))  # , prediction=round(item['pred'].values[0],4))
    tmp_explanation = explanation_conversion(explanation_df, item, explainer)

    # sort values
    sorting_index = item.drop(exclude_attrs, 1).columns
    tmp_explanation = tmp_explanation.sort_values(by='position')
    tmp_explanation = tmp_explanation.sort_values(by=['column', 'position'], key=lambda col: col.map(
        dict(zip(sorting_index, range(len(sorting_index))))))
    res_dict['landmark'] = ast.literal_eval(tmp_explanation.round(4).to_json(orient='records'))
    res_dict['metrics'] = compute_metrics(
        pd.concat([tmp_explanation.score_right_landmark, tmp_explanation.score_left_landmark]))
    return res_dict


def api_adversarial_words(item, impacts_df, exclude_attrs=['id', 'left_id', 'right_id', 'label', 'pred']):
    pred = round(float(item.pred.values[0]),4)
    res_df = dict(content='', label=int(item.label.values[0]),prediction=pred, id=int(item.id.values[0]), add=[])
    cols = np.setdiff1d(item.columns, exclude_attrs)

    record_left = ' | '.join(item[[col for col in cols if col.startswith('left_')]].astype(str).values[0])
    record_right = ' | '.join(item[[col for col in cols if col.startswith('right_')]].astype(str).values[0])
    res_df['content'] = record_left + ' || ' + record_right
    offset_map = {}
    offset = 0
    for col in cols:
        offset_map[col] = offset
        offset += len(item[col].astype(str).values[0].split(' ')) + 1
    add_values = []
    for i in range(impacts_df.shape[0]):
        pos = impacts_df.position.values[i]
        col = impacts_df.column.values[i]
        add_values.append(dict(position=int(pos + offset_map[col]), type=-1, prediction= round(pred- impacts_df.impact.values[i],4)))
    res_df.update(add=add_values)
    return res_df

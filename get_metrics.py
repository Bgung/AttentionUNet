import wandb

import numpy as np
import pandas as pd

from tqdm import tqdm

def get_best_epoch_metrics_dataframe(project_name, run_id, entity=None, metric_names=None):
    """
    wandb run에서 val-loss가 가장 낮은 epoch와 해당 epoch의 지정된 메트릭들을 DataFrame으로 반환합니다.

    Args:
        project_name: wandb 프로젝트 이름 (예: "my-awesome-project").
        run_id: wandb run ID.
        entity: wandb entity (user or team), optional. None이면 현재 사용자의 entity를 사용합니다.
        metric_names: 가져올 메트릭 이름 목록 (예: ["val_loss", "accuracy"]). None이면 모든 메트릭을 가져옵니다.

    Returns:
        val_loss가 가장 낮은 epoch와 해당 epoch의 메트릭들이 포함된 Pandas DataFrame,
        또는 run을 찾을 수 없거나 history가 없는 경우 None.
        만약 step이 logging되지 않았다면, epoch 번호로 동작합니다.
        metric_names가 주어지고, history에 해당 metric이 없을 경우, 해당 metric은 dataframe에 포함되지 않습니다.
    """

    api = wandb.Api()
    full_run_path = (entity + "/" if entity else "") + project_name + "/" + run_id
    try:
        run = api.run(full_run_path)
        run_name = run.name
    except wandb.CommError:
        print(f"Error: Run {full_run_path} not found.")
        return None

    history = run.history(keys=["val-loss", "_step"], pandas=False)

    if not history:
        print(f"Warning: No history found for run {run_id}. Trying epoch instead of _step.")
        history = run.history(keys=["val-loss", "epoch"], pandas=False)
        if not history:
            print(f"Warning: No epoch or step found for run {run_id}. Returning None.")
            return None

    best_epoch = None
    min_val_loss = float('inf')

    for item in history:
        if "val-loss" in item and item["val-loss"] is not None:
            val_loss = item["val-loss"]
            step = item.get("_step") or item.get("epoch")
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_epoch = step

    if best_epoch is None:
        print(f"Warning: No valid val_loss found for run {run_id}. Returning None.")
        return None
        
    metrics_data = {"run_name": [run_name]}
    if metric_names is None:
      all_metrics = run.summary.keys()
      metric_names = list(all_metrics)

    for metric_name in metric_names:
        if metric_name == "val_loss":
            metrics_data[metric_name] = [min_val_loss]
            continue

        metric_history = run.history(keys=[metric_name, "_step"], pandas=False)
        if not metric_history:
            metric_history = run.history(keys=[metric_name, "epoch"], pandas=False)
            if not metric_history:
              print(f"Warning: metric {metric_name} not found in history. Skipping")
              continue

        for item in metric_history:
            metric_step = item.get("_step") or item.get("epoch")

            if metric_step == best_epoch:
                metrics_data[metric_name] = [item.get(metric_name)]
                break
    if not metrics_data: #metric data가 없는경우 빈 dataframe 반환
        return pd.DataFrame()
        
    df = pd.DataFrame(metrics_data)
    return df

def get_best_val_metrics(project_name, run_id, entity=None, metric_names=None):
    """
    wandb에서 각 메트릭의 최소/최대 val 값을 찾아 pandas DataFrame으로 반환합니다.

    Args:
        project_name (str): wandb 프로젝트 이름
        run (str): wandb run 이름
        entity (str, optional): wandb entity. Defaults to None.
        metric_names (list, optional): 찾을 메트릭 이름 리스트. Defaults to None.

    Returns:
        pandas.DataFrame: 각 메트릭의 최소 val 값을 담은 DataFrame
    """

    api = wandb.Api()
    full_run_path = (entity + "/" if entity else "") + project_name + "/" + run_id
    try:
        run = api.run(full_run_path)
        run_name = run.name
    except wandb.CommError:
        print(f"Error: Run {full_run_path} not found.")
        return None

    best_values = {'run_name': run_name}
    for metric in metric_names:
        metric_history = run.history(keys=[metric, 'epoch'], pandas=False)
        if not metric_history:
            print(f"Warning: metric {metric} not found in history. Skipping")
            continue
        if metric.lower() in ['train-miou', 'val-miou', 'train-dice', 'val-dice']:
            best_value = max(item.get(metric) for item in metric_history if item.get(metric) is not None)
        else:
            best_value = min(item.get(metric) for item in metric_history if item.get(metric) is not None)
        best_values[metric] = best_value


    # DataFrame 생성 및 형식 지정
    df = pd.DataFrame(best_values, index=[0])

    return df


def get_sample(project_name, run_id, entity=None):
    api = wandb.Api()
    full_run_path = (entity + "/" if entity else "") + project_name + "/" + run_id
    try:
        run = api.run(full_run_path)
        run_name = run.name
    except wandb.CommError:
        print(f"Error: Run {full_run_path} not found.")
        return None

    sample = {'run_name': run_name}
    
    for k, v in run.summary.items():
        if isinstance(v, dict) and "path" in v and "format" in v:  # wandb.Image 확인
            if v["format"] in ("png", "jpeg", "jpg"):
                print(f"Image Key: {k}")
                image_url = v["path"]
                print(f"Image URL: {image_url}")



runs = {
    "EHBI": [
        'a8cpw2g8',
        't7ftqo63',
        'mpe6oti7',
        'ynh630sd',
        'lfh56a22',
        'orn6egq8'
    ],
    "KVASIR": [
        '2rh3xoe3',
        '1t8nmfid',
        'nh8cgh7q',
        'ub37o3u0',
        '9jxutlnh',
        'z85vi3gw'
    ],
    "HRF": [
        'f22xi5oj',
        '3pb1vp9t',
        'y71omy78',
        '1pacgpq1',
        '84o1cey4',
        'qz79ljov'
    ]
}

# 사용 예시
project_name = "MedicalAI2024"  # 실제 프로젝트 이름으로 변경
entity = None  # optional
metric_names = [
    "train-loss",
    "train-seg_loss",
    "train-ctl_loss",
    "val-loss",
    "val-seg_loss",
    "val-ctl_loss",
    "train-Dice",
    "train-mIoU",
    "val-Dice",
    "val-mIoU",
] # optional

for dataset, run_ids in runs.items():
    dfs = []
    for run_id in tqdm(run_ids):
        get_sample(project_name, run_id, entity)
        # df = get_best_epoch_metrics_dataframe(project_name, run_id, entity, metric_names)
    #     df = get_best_val_metrics(project_name, run_id, entity, metric_names)
        
    #     if df is not None and df.empty:
    #         print("No metric data found")
    #         continue

    #     dfs.append(df)
    # df_all = pd.concat(dfs, ignore_index=True)

    # df_all.to_csv(f"medicalai2024_metrics_{dataset}.csv", index=False)

# for dataset in runs.keys():
#     print(dataset)
#     df_all = pd.read_csv(f"medicalai2024_metrics_{dataset}.csv", index_col=False)
#     df_all = df_all[['run_name', 'val-loss', 'val-seg_loss', 'val-ctl_loss', 'val-mIoU', 'val-Dice']]
#     df_all = df_all.T
#     df_all.columns = df_all.iloc[0]
#     df_all = df_all.drop(df_all.index[0])
#     df_all = df_all[['UNetPLSEG', 'AttentionUNetPLSEG', 'MHAUNetPLSEG', 'UNetPLCTL', 'AttentionUNetPLCTL', 'MHAUNetPLCTL']]
#     print(df_all)
#     for row in df_all.iterrows():
#         for i, val in enumerate(row[1]):
#             val: float = val
#             if not np.isnan(val):
#                 print(round(float(val), 3), end=' ')
#             else:
#                 print('nan', end=' ')
#         print()
#     df_all.to_html(f"medicalai2024_metrics_{dataset}.html")

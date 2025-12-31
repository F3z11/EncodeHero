import pandas as pd
import warnings
import glob

warnings.filterwarnings("ignore")

# Compute median value for each model-encoder pair
def compute_median_model(data, metric, task):
    """
    Compute the median performance metrics for each encoding method across different models.

    Parameters:
    - data (pd.DataFrame): The input dataset containing model performance metrics.
    - metric (str): The performance metric to compute the median for (e.g., 'f1', 'rmse').
    - task (str): The type of machine learning task ('binary', 'multiclass', 'regression').

    Returns:
    - ris (pd.DataFrame): A DataFrame containing the median performance metrics for each encoding method and model.
    
    Raises:
    - ValueError: If the task parameter is not one of 'binary', 'multiclass', or 'regression'.
    """

    if task not in ['binary', 'multiclass', 'regression']:
            raise ValueError(f"Invalid parameter. Expected one of: 'binary', 'multiclass', 'regression'. Got: {task}")

    if task == 'multiclass' and 'woe' in data.columns:
       data = data.drop(columns=['woe'])

    ris = data[data.model == data.model.unique()[0]].groupby(by=['encoding'], as_index=False, sort=False)[metric].median()
    for model in data.model.unique()[1:]:
        ris[model] = data[data.model == model].groupby(by=['encoding'], as_index=False, sort=False)[metric].median()[metric]
    ris.columns = ['encoding', 'nb', 'RF', 'mlp', 'knn', 'svm', 'lgr', 'dt']
    ris = round(ris, 3)

    # Here we also need to save these results on the database

    return ris

# Compute the best encoder for each model with the associated metric value (used to show to the user)
def return_best_encoders(data):
  """
    Identify the best encoding method for each model based on the highest numeric value in the dataset.

    Parameters:
    - data (pd.DataFrame): The input DataFrame where rows represent encoding methods and columns represent models or metrics.
                           The DataFrame should contain numeric values representing performance metrics.

    Returns:
    - res (pd.DataFrame): A DataFrame containing the best encoding method for each model. The DataFrame includes:
        - 'encoding': The encoding method with the highest value for each model.
        - 'model': The name of the model or metric associated with the highest value.
        - 'value': The maximum value for each model.
    """
  indexes = data.idxmax(numeric_only=True)
  res = pd.DataFrame(data.iloc[indexes]['encoding'])
  res['model'] = data.max(numeric_only=True).index.tolist()
  res['value'] = data.max(numeric_only=True).values.tolist()

  return res

# Compute the median of each encoder for all models (used for graphical tool)
def compute_median_all(data, metric, task):
  """
    Compute the median performance metric for each encoding method across different models.

    Parameters:
    - data (pd.DataFrame): The input dataset containing model performance metrics, 
                           where rows represent different encodings and columns represent different models or metrics.
    - metric (str): The performance metric to compute the median for.
    - task (str): The type of machine learning task. Must be one of 'binary', 'multiclass', or 'regression'.

    Returns:
    - ris (pd.DataFrame): A DataFrame containing the median performance metric for each encoding method,
                          sorted by the specified metric.

    Raises:
    - ValueError: If the 'task' parameter is not one of 'binary', 'multiclass', or 'regression'.
    """
  if task not in ['binary', 'multiclass', 'regression']:
        raise ValueError(f"Invalid parameter. Expected one of: 'binary', 'multiclass', 'regression'. Got: {task}")

  if task == 'multiclass' and 'woe' in data.columns:
    df = df.drop(columns=['woe'])

  ris = data.groupby(by=['encoding'], as_index=False, sort=False)[metric].median().sort_values(by=[metric])
  
  return ris

#TODO: finish to setup this function

def return_results(metric, task):
    """
    Load and aggregate performance metrics from multiple CSV files, then compute the median metric 
    for each encoding method based on the specified machine learning task.

    Parameters:
    - metric (str): The performance metric to compute the median for.
    - task (str): The type of machine learning task. Must be one of 'binary', 'multiclass', or 'regression'.

    Returns:
    - ris (pd.DataFrame): A DataFrame containing the median performance metric for each encoding method, 
                          aggregated across all input files and sorted by the specified metric.
    """
    if task not in ['binary', 'multiclass', 'regression']:
            raise ValueError(f"Invalid parameter. Expected one of: 'binary', 'multiclass', 'regression'. Got: {task}")

    if task in ['binary', 'multiclass']:
      if metric not in ['f1', 'precision', 'recall']:
            raise ValueError(f"Invalid parameter. Expected one of: 'f1', 'precision', 'recall'. Got: {task}")
      
      filepaths = glob.glob('results/class/*.csv')
    else:
      if metric not in ['mae', 'mse', 'rmse', 'r2']:
            raise ValueError(f"Invalid parameter. Expected one of: 'f1', 'precision', 'recall'. Got: {task}")
      
      filepaths = glob.glob('results/regr/*.csv') 

    all_dfs = [pd.read_csv(fp) for fp in filepaths]
    data = pd.concat(all_dfs, ignore_index=True)

    ris = compute_median_all(data, metric, task)

    return ris
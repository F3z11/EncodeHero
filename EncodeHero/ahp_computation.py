import ahpy
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
try:
    # Python 3.9+ standard library
    from importlib import resources as importlib_resources
except Exception:
    # backport (if present) for older Python versions
    import importlib_resources  # type: ignore

from pathlib import Path as _Path


def resource_filename_compat(package: str, resource: str) -> str:
    """Return a filesystem path to a package resource.

    Uses importlib.resources (Python 3.9+) or falls back to a relative path.
    """
    # Try importlib.resources.files -> Traversable
    try:
        files = importlib_resources.files(package)
        candidate = files.joinpath(resource)
        # Try to return a string path for on-disk resources
        try:
            return str(candidate)
        except Exception:
            # Continue to fallback
            pass
    except Exception:
        pass

    # Fallback: assume package exists as a sibling directory in source tree
    base = _Path(__file__).resolve().parent
    guess = base.parent / package / resource
    return str(guess)

def get_results_path() -> Path:
    """Get the path to the results folder."""
    try:
        files = importlib_resources.files('EncodeHero')
        return Path(str(files.joinpath('results')))
    except Exception:
        return _Path(__file__).resolve().parent / 'results'

def load_aggregated_data(task: str, model: str = 'all') -> dict:
    """Load and aggregate all datasets and seeds for a given task.
    
    Args:
        task: One of 'binary', 'multiclass', or 'regression'
        model: Model name to filter by, or 'all' for aggregation across all models
    
    Returns:
        Dictionary mapping metric names to encoder->value dictionaries
    """
    results_path = get_results_path()
    task_path = results_path / task
    
    # Map display model names to actual model names in CSV
    model_name_map = {
        'RF': 'RandomForestClassifier',
        'RandomForest': 'RandomForestClassifier',
        'SVM': 'SVC',
        'svm': 'SVC',
        'MLP': 'MLPClassifier',
        'mlp': 'MLPClassifier',
        'K-NN': 'KNeighborsClassifier',
        'knn': 'KNeighborsClassifier',
        'LNR': 'LinearRegression',
        'lgr': 'LogisticRegression',
        'DT': 'DecisionTreeClassifier',
        'dt': 'DecisionTreeClassifier',
        'nb': 'GaussianNB'
    }
    
    # For regression, adjust model names
    if task == 'regression':
        model_name_map.update({
            'RF': 'RandomForestRegressor',
            'SVM': 'SVR',
            'MLP': 'MLPRegressor',
            'K-NN': 'KNeighborsRegressor',
            'LNR': 'LinearRegression',
            'DT': 'DecisionTreeRegressor'
        })
    
    # Map encoder names from CSV to standardized names
    encoder_name_map = {
        'BackwardDifferenceEncoder': 'BackwardDifference',
        'BaseNEncoder': 'BaseN',
        'BinaryEncoder': 'Binary',
        'CatBoostEncoder': 'CatBoost',
        'CountEncoder': 'Count',
        'GLMMEncoder': 'GLMM',
        'GrayEncoder': 'Gray',
        'HashingEncoder': 'Hashing',
        'HelmertEncoder': 'Helmert',
        'JamesSteinEncoder': 'JamesStein',
        'LeaveOneOutEncoder': 'LeaveOneOut',
        'MEstimateEncoder': 'MEstimate',
        'OneHotEncoder': 'OneHot',
        'OrdinalEncoder': 'Ordinal',
        'PolynomialEncoder': 'Polynomial',
        'RankHotEncoder': 'RankHot',
        'SumEncoder': 'Sum',
        'TargetEncoder': 'Target',
        'MinHashEncoder': 'MinHash',
        'SimilarityEncoder': 'Similarity',
        'GapEncoder': 'Gap',
        'MeanEncoder': 'Mean',
        'DecisionTreeEncoder': 'DecisionTree',
        'WOEEncoder': 'WOE'
    }
    
    # Get all result files for this task
    result_files = glob.glob(str(task_path / 'results_*.csv'))
    
    if not result_files:
        raise ValueError(f"No result files found in {task_path}")
    
    # Load and concatenate all files
    all_data = []
    for file in result_files:
        df = pd.read_csv(file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Filter by model if not 'all'
    if model != 'all':
        filter_model = model_name_map.get(model, model)
        combined_df = combined_df[combined_df['model'] == filter_model]
    
    # Map encoder names to standardized names
    combined_df['encoding'] = combined_df['encoding'].map(encoder_name_map).fillna(combined_df['encoding'])
    
    # Determine which metrics to use based on task
    if task == 'regression':
        metrics = ['mae', 'mse', 'rmse', 'r2']
    else:  # binary or multiclass
        metrics = ['acc', 'prec', 'recall', 'f1', 'auc', 'mcc']
    
    # Aggregate by encoder - compute mean across all datasets and seeds
    result = {}
    for metric in metrics:
        if metric in combined_df.columns:
            grouped = combined_df.groupby('encoding')[metric].median()
            # Filter out invalid values (negative, NaN, inf)
            # AHP requires all values to be positive
            grouped = grouped.replace([np.inf, -np.inf], np.nan)
            grouped = grouped.dropna()
            # For metrics where lower is better (mae, mse, rmse), keep as is
            # For metrics where higher is better but might be negative (r2), ensure positive
            if metric == 'r2':
                # R² can be negative; shift to positive range if needed
                min_val = grouped.min()
                if min_val < 0:
                    grouped = grouped - min_val + 0.001  # Shift to positive
            # Ensure all values are positive
            if (grouped <= 0).any():
                grouped = grouped[grouped > 0]
            result[metric] = grouped.to_dict()
    
    return result

def load_time_data() -> dict:
    """Load and aggregate time data across all datasets and seeds.
    
    Returns:
        Dictionary mapping encoder names to average time values
    """
    results_path = get_results_path()
    time_path = results_path / 'times'
    
    # Map encoder names from CSV to standardized names
    encoder_name_map = {
        'BackwardDifferenceEncoder': 'BackwardDifference',
        'BaseNEncoder': 'BaseN',
        'BinaryEncoder': 'Binary',
        'CatBoostEncoder': 'CatBoost',
        'CountEncoder': 'Count',
        'GLMMEncoder': 'GLMM',
        'GrayEncoder': 'Gray',
        'HashingEncoder': 'Hashing',
        'HelmertEncoder': 'Helmert',
        'JamesSteinEncoder': 'JamesStein',
        'LeaveOneOutEncoder': 'LeaveOneOut',
        'MEstimateEncoder': 'MEstimate',
        'OneHotEncoder': 'OneHot',
        'OrdinalEncoder': 'Ordinal',
        'PolynomialEncoder': 'Polynomial',
        'RankHotEncoder': 'RankHot',
        'SumEncoder': 'Sum',
        'TargetEncoder': 'Target',
        'MinHashEncoder': 'MinHash',
        'SimilarityEncoder': 'Similarity',
        'GapEncoder': 'Gap',
        'MeanEncoder': 'Mean',
        'DecisionTreeEncoder': 'DecisionTree',
        'WOEEncoder': 'WOE'
    }
    
    # Get all time files
    time_files = glob.glob(str(time_path / 'time_*.csv'))
    
    if not time_files:
        raise ValueError(f"No time files found in {time_path}")
    
    # Load and concatenate all files
    all_data = []
    for file in time_files:
        df = pd.read_csv(file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Map encoder names to standardized names
    combined_df['encoder'] = combined_df['encoder'].map(encoder_name_map).fillna(combined_df['encoder'])
    
    # Aggregate by encoder - compute mean across all datasets and seeds
    grouped = combined_df.groupby('encoder')['time'].median()
    
    # Filter out invalid values (negative, NaN, inf)
    grouped = grouped.replace([np.inf, -np.inf], np.nan)
    grouped = grouped.dropna()
    # Ensure all values are positive
    grouped = grouped[grouped > 0]
    
    return grouped.to_dict()

class BestEncoder:
    def __init__(self, task, model, comparisons):

        self.task = task
        self.model = model
        self.report = None
        
        # Create dictionaries to store user inputs for each group
        self.comparisons = comparisons

        # Load aggregated performance data
        perf_data = load_aggregated_data(task, model)
        
        # Load time data
        time_data = load_time_data()
        
        # Map metric names from CSV to display names
        metric_name_map = {
            'acc': 'Accuracy',
            'prec': 'Precision',
            'recall': 'Recall',
            'f1': 'F1',
            'auc': 'AUC',
            'mcc': 'MCC'
        }
        
        # Store measured values as dictionaries (encoder -> value)
        self.measured_values = {}
        
        # Map performance metrics to display names
        for csv_metric, display_metric in metric_name_map.items():
            if csv_metric in perf_data:
                self.measured_values[display_metric] = perf_data[csv_metric]
        
        # For regression, use lowercase metric names
        for metric in ['mae', 'mse', 'rmse', 'r2']:
            if metric in perf_data:
                self.measured_values[metric] = perf_data[metric]
        
        # Add time data
        self.measured_values['Time'] = time_data

        self.performance_values_map = {
                'multiclass': ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC'],
                'binary': ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC'],
                'regression': ['mae', 'mse', 'rmse', 'r2']
        }

        self.calculate()

    def calculate(self):
        """
        Performs Analytic Hierarchy Process (AHP) calculations based on various criteria.

        This method normalizes measured values for different criteria (time and performance metrics),
        creates comparisons, establishes a hierarchy, and generates a criteria report using the AHP methodology.

        The function uses the 'ahpy' library for AHP calculations.

        Raises:
            Exception: Any exception that occurs during the calculation process is caught,
                    its traceback is printed, and the error is re-raised.

        Returns:
            None. Sets self.report with the AHP analysis results.
        """
        try:
            performance_values = self.performance_values_map[self.task]
            
            # Build data dict from measured values
            # First, find common encoders across all metrics to ensure consistency
            all_metrics = performance_values + ['Time']
            encoder_sets = []
            for metric in all_metrics:
                if metric in self.measured_values:
                    encoder_sets.append(set(self.measured_values[metric].keys()))
            
            # Find intersection of all encoder sets
            if encoder_sets:
                common_encoders = encoder_sets[0].intersection(*encoder_sets[1:])
            else:
                raise ValueError("No valid data found for any metrics")
            
            if not common_encoders:
                raise ValueError("No encoders have valid data for all metrics")
            
            # Build data dict with only common encoders
            data = {}
            for key in all_metrics:
                if key in self.measured_values:
                    # Filter to only include common encoders
                    filtered_data = {enc: val for enc, val in self.measured_values[key].items() 
                                   if enc in common_encoders and val > 0}
                    
                    if filtered_data:
                        # For metrics where LOWER is BETTER, invert first
                        if key in ['Time', 'mae', 'mse', 'rmse']:
                            # Invert: use reciprocal so lower values become higher
                            filtered_data = {enc: 1.0/val for enc, val in filtered_data.items()}
                        
                        # Apply Min-Max normalization to scale all metrics to [0, 1]
                        # This ensures all metrics have the same scale and influence
                        values = list(filtered_data.values())
                        min_val = min(values)
                        max_val = max(values)
                        
                        if max_val > min_val:  # Avoid division by zero
                            # Normalize to [0.01, 1.0] range (avoid exact 0 for AHP)
                            normalized_data = {
                                enc: 0.01 + 0.99 * (val - min_val) / (max_val - min_val)
                                for enc, val in filtered_data.items()
                            }
                            data[key] = normalized_data
                        else:
                            # All values are the same, assign equal normalized value
                            data[key] = {enc: 0.5 for enc in filtered_data.keys()}
            
            if not data:
                raise ValueError("No valid data available after filtering")
            
            compare_list = {}
            children_perf = []
            
            # Create comparison objects for each metric
            for key, value in data.items():
                comp = ahpy.Compare(key, value, precision=3)
                compare_list[f"{key}_value"] = comp
                if key in performance_values:
                    children_perf.append(comp)

            # Create comparison objects from user inputs
            for group, comparisons in self.comparisons.items():
                if comparisons:  # Only create if there are comparisons
                    comparison_data = {k: v["value"] if v["selected_criteria"] == k[0] else 1/v["value"] for k, v in comparisons.items()}
                    compare_list[group] = ahpy.Compare(group, comparison_data, precision=3)

            # Build hierarchy
            # Since Cost only has Time, we create a simple comparison for Cost with just Time
            # This means Cost weight = Time weight at the Cost level
            compare_list['Cost'] = ahpy.Compare('Cost', {'Time': 1.0}, precision=3)
            compare_list['Cost'].add_children([compare_list["Time_value"]])
            compare_list['Performance'].add_children(children_perf)

            compose = ahpy.Compose()
            compose.add_comparisons(list(compare_list.values()))
            hierarchy = {
                'Criteria': ['Performance', 'Cost'],
                'Performance': performance_values,
                'Cost': ['Time']
            }
            compose.add_hierarchy(hierarchy)

            self.report = compose.report()
        except Exception as e:
            print(traceback.format_exc())
            print("Error:", e)
            # Re-raise the exception so the caller knows the calculation failed
            raise
    
    def get_report(self):
        """
        Returns the calculated target weights.
        
        Returns:
            dict: The calculated target weights.
        """
        return self.report
    
    def ranking_plot(res: dict):
        """
        Generate a horizontal bar plot ranking the encoders based on their priority values.

        Parameters:
        - res (dict): A dictionary where keys are encoder names and values are their corresponding priority values.

        Returns:
        - fig: A Matplotlib figure object representing the generated plot.
        """

        keys = list(res.keys())
        values = list(res.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        labels = list(sorted_dict.keys())
        values = list(sorted_dict.values())

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(labels, values, color='skyblue')
        ax.set_xlabel('Priority value')
        ax.set_ylabel('Encoder')
        ax.set_title('Priority of each encoder')
        
        return fig

    @classmethod
    def get_ranking(cls, task:str, model:str, comparisons:dict):
        """
        Get the ranking report based on the task and comparison data provided.

        Parameters:
        - task (str): The type of machine learning task. Expected values are 'binary', 'multiclass', or 'regression'.
        - comparisons (dict): A dictionary containing comparison data for generating the ranking.

        Returns:
        - The ranking report generated by the application.
        
        Raises:
        - ValueError: If the task parameter is not one of 'binary', 'multiclass', or 'regression'.
        """

        if task not in ['binary', 'multiclass', 'regression']:
            raise ValueError(f"Invalid parameter. Expected one of: 'binary', 'multiclass', 'regression'. Got: {task}")

        app = cls(task, model, comparisons)

        return app.get_report()
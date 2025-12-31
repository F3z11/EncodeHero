"""Main experiment pipeline for encoder benchmarking"""
import warnings
import os
import time
import math
import traceback
import pandas as pd
import numpy as np
import pickle
import gc
import sklearn
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tqdm import tqdm
from category_encoders import (
    BackwardDifferenceEncoder,
    BaseNEncoder,
    BinaryEncoder,
    CatBoostEncoder,
    CountEncoder,
    GLMMEncoder,
    GrayEncoder,
    HashingEncoder,
    HelmertEncoder,
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    MEstimateEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialEncoder,
    QuantileEncoder,
    RankHotEncoder,
    SummaryEncoder,
    SumEncoder,
    TargetEncoder,
    WOEEncoder,
)
from skrub import MinHashEncoder, SimilarityEncoder, GapEncoder
from feature_engine.encoding import MeanEncoder, DecisionTreeEncoder

try:
    import EncodeHero.res_analysis as res_analysis
except:
    from . import res_analysis

sklearn.set_config(transform_output="default")
warnings.filterwarnings("ignore")


class MLExperiment:
    """Main experiment class for benchmarking categorical encoders"""
    
    def __init__(self, data, target='target', task='binary', seed=42, dataset_name='dataset'):
        """Initialize experiment.
        
        Args:
            data: Input DataFrame
            target: Name of target column
            task: Task type ('binary', 'multiclass', 'regression')
            seed: Random seed for reproducibility
            dataset_name: Name of the dataset (used for folder naming)
        """
        self.data = data
        self.target = target
        self.task = task
        self.seed = seed
        self.dataset_name = dataset_name
        
    def _optimize_data_types(self, df):
        """Optimize DataFrame memory usage"""
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        return df
    
    def _missing_input(self, X, cat_cols, num_cols):
        """Handle missing values in features"""
        if len(num_cols) > 0:
            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            X[num_cols] = pd.DataFrame(
                imp_mean.fit_transform(X[num_cols]), 
                columns=num_cols, 
                index=X.index
            )
        
        if len(cat_cols) > 0:
            imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X[cat_cols] = pd.DataFrame(
                imp_freq.fit_transform(X[cat_cols]), 
                columns=cat_cols, 
                index=X.index
            )
        
        return X
    
    def _preprocess_once(self):
        """Preprocess data once before encoder evaluation"""
        data = self.data.copy()
        
        # Optimize data types
        data = self._optimize_data_types(data)
        
        # Process target for classification
        if self.task in ('binary', 'multiclass'):
            if not pd.api.types.is_numeric_dtype(data[self.target]):
                data[self.target] = pd.factorize(data[self.target])[0]
        
        # Extract features and target
        X = data.drop(columns=[self.target])
        y = data[self.target]
        
        # Identify column types
        cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
        num_cols = X.select_dtypes(exclude=["category", "object"]).columns.tolist()
        
        # Handle missing values
        X = self._missing_input(X, cat_cols, num_cols)
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'cat_cols': cat_cols,
            'num_cols': num_cols,
        }
    
    def _get_encoders(self, cat_cols):
        """Get list of encoders for evaluation"""
        data = self.data
        
        # Compute optimal base for BaseNEncoder
        if len(cat_cols) > 0:
            unique_counts = data[cat_cols].nunique()
            max_card = unique_counts.max()
            dim = {
                b: (unique_counts / b).apply(math.ceil).sum() 
                for b in range(2, min(max_card, 10))
            }
            base = min(dim, key=dim.get) if dim else 2
        else:
            base = 2
        
        # Compute fill_value for DecisionTreeEncoder
        if self.task == 'regression':
            if not pd.api.types.is_numeric_dtype(data[self.target]):
                data[self.target] = pd.to_numeric(data[self.target], errors='coerce')
            fill_value = float(data[self.target].mean())
        else:
            fill_value = 0.0
        
        encoders = [
            BackwardDifferenceEncoder(),
            BaseNEncoder(base=base),
            BinaryEncoder(),
            CatBoostEncoder(random_state=self.seed),
            CountEncoder(),
            GLMMEncoder(random_state=self.seed),
            GrayEncoder(),
            HashingEncoder(),
            HelmertEncoder(),
            JamesSteinEncoder(random_state=self.seed),
            LeaveOneOutEncoder(random_state=self.seed),
            MEstimateEncoder(random_state=self.seed),
            OneHotEncoder(),
            OrdinalEncoder(),
            PolynomialEncoder(),
            RankHotEncoder(),
            SumEncoder(),
            TargetEncoder(),
            MinHashEncoder(),
            SimilarityEncoder(),
            GapEncoder(ngram_range=(1, 4), random_state=self.seed),
            MeanEncoder(unseen='encode'),
            DecisionTreeEncoder(
                regression=(self.task == 'regression'), 
                unseen='encode',
                fill_value=fill_value, 
                random_state=self.seed
            )
        ]
        
        # Add task-specific encoders
        if self.task == 'regression':
            encoders.extend([QuantileEncoder(), SummaryEncoder()])
        elif self.task == 'binary':
            encoders.append(WOEEncoder(random_state=self.seed))
        
        return encoders
    
    def _get_encoder_param_grids(self):
        """Get parameter grids for tunable encoders"""
        return {
            'HashingEncoder': {
                'preprocessor__encoder__n_components': [4, 8, 16],
                'preprocessor__encoder__hash_method': ['md5', 'sha256']
            },
            'CatBoostEncoder': {
                'preprocessor__encoder__a': [0.5, 1, 2],
                'preprocessor__encoder__sigma': [0.01, 0.05, 0.1]
            },
            'JamesSteinEncoder': {
                'preprocessor__encoder__sigma': [0.01, 0.05, 0.1]
            },
            'LeaveOneOutEncoder': {
                'preprocessor__encoder__sigma': [0.01, 0.05, 0.1]
            },
            'MEstimateEncoder': {
                'preprocessor__encoder__sigma': [0.01, 0.05, 0.1],
                'preprocessor__encoder__m': [0, 1, 2]
            },
            'TargetEncoder': {
                'preprocessor__encoder__min_samples_leaf': [10, 20, 30],
                'preprocessor__encoder__smoothing': [5, 10, 20]
            }
        }
    
    def _get_no_tune_encoders(self):
        """Get list of encoders that don't need hyperparameter tuning"""
        return [
            'BackwardDifferenceEncoder', 'BaseNEncoder', 'BinaryEncoder',
            'CountEncoder', 'GrayEncoder', 'HelmertEncoder', 'OneHotEncoder',
            'OrdinalEncoder', 'PolynomialEncoder', 'QuantileEncoder',
            'RankHotEncoder', 'SummaryEncoder', 'SumEncoder', 'WOEEncoder',
            'MinHashEncoder', 'SimilarityEncoder', 'GapEncoder',
            'MeanEncoder', 'DecisionTreeEncoder', 'GLMMEncoder'
        ]
    
    def _evaluate_classification(self, model, X_test, y_test, task):
        """Evaluate classification model"""
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0, average='macro')
        recall = recall_score(y_test, y_pred, zero_division=0, average='macro')
        f1 = f1_score(y_test, y_pred, zero_division=0, average='macro')
        
        # AUC
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if task == 'binary':
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            else:
                auc = 0.0
        except:
            auc = 0.0
        
        # MCC
        try:
            mcc = matthews_corrcoef(y_test, y_pred)
        except:
            mcc = 0.0
        
        return [acc, prec, recall, f1, auc, mcc]
    
    def _evaluate_regression(self, model, X_test, y_test):
        """Evaluate regression model"""
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        
        return [mae, mse, rmse, r2]
    
    def _evaluate_encoder(self, encoder, preprocessed_data, cache_dir=None):
        """Evaluate a single encoder with optional hyperparameter tuning"""
        X_train = preprocessed_data['X_train']
        X_test = preprocessed_data['X_test']
        y_train = preprocessed_data['y_train']
        y_test = preprocessed_data['y_test']
        cat_cols = preprocessed_data['cat_cols']
        
        no_tune_encoders = self._get_no_tune_encoders()
        tune_encoders = self._get_encoder_param_grids()
        
        # Select models based on task
        if self.task in ('binary', 'multiclass'):
            models = [
                RandomForestClassifier(random_state=self.seed),
                SVC(random_state=self.seed, probability=True),
                LogisticRegression(random_state=self.seed, max_iter=1000),
                GaussianNB(),
                MLPClassifier(random_state=self.seed, early_stopping=True, max_iter=1000),
                KNeighborsClassifier(),
                DecisionTreeClassifier(random_state=self.seed)
            ]
            metric_columns = ['model', 'acc', 'prec', 'recall', 'f1', 'auc', 'mcc']
        else:
            models = [
                RandomForestRegressor(random_state=self.seed),
                SVR(),
                LinearRegression(),
                MLPRegressor(random_state=self.seed, early_stopping=True, max_iter=1000),
                KNeighborsRegressor(),
                DecisionTreeRegressor(random_state=self.seed)
            ]
            metric_columns = ['model', 'mae', 'mse', 'rmse', 'r2']
        
        results_list = []
        encoder_name = type(encoder).__name__
        
        try:
            start = time.time()
            
            if encoder_name in no_tune_encoders:
                # Handle special encoders
                if isinstance(encoder, (MinHashEncoder, GapEncoder)):
                    catpipe = [(f'enc_{col_name}', encoder, col_name) for col_name in cat_cols]
                    preprocessor = ColumnTransformer(catpipe, remainder='passthrough')
                else:
                    preprocessor = ColumnTransformer([
                        ('encoder', encoder, cat_cols)
                    ], remainder='passthrough')
                
                # Apply preprocessing
                X_train_transformed = preprocessor.fit_transform(X_train, y_train)
                X_test_transformed = preprocessor.transform(X_test)
                
                # Apply scaling
                scaler = StandardScaler()
                X_train_transformed = scaler.fit_transform(X_train_transformed)
                X_test_transformed = scaler.transform(X_test_transformed)
                
                # Evaluate models
                for model in models:
                    model.fit(X_train_transformed, y_train)
                    
                    if self.task in ('binary', 'multiclass'):
                        metrics = self._evaluate_classification(model, X_test_transformed, y_test, self.task)
                    else:
                        metrics = self._evaluate_regression(model, X_test_transformed, y_test)
                    
                    results_list.append([type(model).__name__, *metrics])
            else:
                # For tunable encoders, use hyperparameter tuning
                for model in models:
                    pipeline = Pipeline([
                        ('preprocessor', ColumnTransformer([
                            ('encoder', encoder, cat_cols)
                        ], remainder='passthrough')),
                        ('scaler', StandardScaler()),
                        ('model', model)
                    ])
                    
                    param_grid = tune_encoders.get(encoder_name, {})
                    
                    if not param_grid:
                        pipeline.fit(X_train, y_train)
                        
                        if self.task in ('binary', 'multiclass'):
                            metrics = self._evaluate_classification(pipeline, X_test, y_test, self.task)
                        else:
                            metrics = self._evaluate_regression(pipeline, X_test, y_test)
                        
                        results_list.append([type(model).__name__, *metrics])
                        continue
                    
                    # Perform randomized search
                    grid_search = RandomizedSearchCV(
                        estimator=pipeline,
                        param_distributions=param_grid,
                        scoring='f1_macro' if self.task in ('binary', 'multiclass') else 'neg_root_mean_squared_error',
                        n_iter=5,
                        cv=5,
                        random_state=self.seed,
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train, y_train)
                    best_pipeline = grid_search.best_estimator_
                    
                    if self.task in ('binary', 'multiclass'):
                        metrics = self._evaluate_classification(best_pipeline, X_test, y_test, self.task)
                    else:
                        metrics = self._evaluate_regression(best_pipeline, X_test, y_test)
                    
                    results_list.append([type(model).__name__, *metrics])
            
            time_elapsed = time.time() - start
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR evaluating {encoder_name}")
            print(f"{'='*80}")
            traceback.print_exc()
            print(f"{'='*80}\n")
            # Return empty results on error
            time_elapsed = 0
            for model in models:
                if self.task in ('binary', 'multiclass'):
                    results_list.append([type(model).__name__, 0, 0, 0, 0, 0, 0])
                else:
                    results_list.append([type(model).__name__, 0, 0, 0, 0])
        
        results = pd.DataFrame(results_list, columns=metric_columns)
        return results, time_elapsed
    
    def run_experiment(self, export=True, dir=None, use_cache=True):
        """Run the full experiment pipeline.
        
        Args:
            export: Whether to export results to CSV files
            dir: Directory to save results (default: current directory)
            use_cache: Whether to use cached results
            
        Returns:
            dict: Dictionary containing all results
        """
        if dir is None:
            dir = os.getcwd()
        
        # Create results directory with dataset_name_seed format
        results_dir = os.path.join(dir, f"{self.dataset_name}_{self.seed}")
        os.makedirs(results_dir, exist_ok=True)
        
        total_start = time.time()
        
        # Preprocess data once
        print("Preprocessing data...")
        preprocessed_data = self._preprocess_once()
        
        # Define output columns
        if self.task in ('binary', 'multiclass'):
            res_columns = ['encoding', 'model', 'acc', 'prec', 'recall', 'f1', 'auc', 'mcc']
        else:
            res_columns = ['encoding', 'model', 'mae', 'mse', 'rmse', 'r2']
        
        # Get encoders
        cat_cols = preprocessed_data['cat_cols']
        encoders = self._get_encoders(cat_cols)
        
        # Lists for results
        results_rows = []
        times_rows = []
        
        # Create cache directory if using cache
        cache_dir = None
        if use_cache:
            cache_dir = os.path.join(results_dir, "cache", f"{self.task}_{self.seed}")
            os.makedirs(cache_dir, exist_ok=True)
        
        # Evaluate encoders
        print("Evaluating encoders...")
        for encoder in tqdm(encoders, desc="Processing encoders"):
            encoder_name = type(encoder).__name__
            
            # Check cache
            if use_cache and cache_dir:
                cache_file = os.path.join(cache_dir, f"{encoder_name}.pkl")
                
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'rb') as f:
                            res_encoder, time_elapsed = pickle.load(f)
                    except Exception:
                        print(f"Error loading cache for {encoder_name}, re-evaluating...")
                        res_encoder, time_elapsed = self._evaluate_encoder(encoder, preprocessed_data)
                        with open(cache_file, 'wb') as f:
                            pickle.dump((res_encoder, time_elapsed), f)
                else:
                    res_encoder, time_elapsed = self._evaluate_encoder(encoder, preprocessed_data)
                    with open(cache_file, 'wb') as f:
                        pickle.dump((res_encoder, time_elapsed), f)
            else:
                res_encoder, time_elapsed = self._evaluate_encoder(encoder, preprocessed_data)
            
            # Record results
            for _, row in res_encoder.iterrows():
                results_rows.append([encoder_name] + row.tolist())
            
            times_rows.append([encoder_name, time_elapsed])
            
            # Force garbage collection
            gc.collect()
        
        # Create DataFrames from collected rows
        print("Creating result DataFrames...")
        results_df = pd.DataFrame(results_rows, columns=res_columns)
        times_df = pd.DataFrame(times_rows, columns=['encoder', 'time'])
        
        # Prepare final results
        final_res = {
            'res_raw': results_df,
            'time_dim': times_df
        }
        
        # Results elaboration
        if self.task in ('binary', 'multiclass'):
            final_res['model_median'] = res_analysis.compute_median_model(results_df, 'f1', self.task)
            final_res['all_median'] = res_analysis.compute_median_all(results_df, 'f1', self.task)
        else:
            final_res['model_median'] = res_analysis.compute_median_model(results_df, 'rmse', self.task)
            final_res['all_median'] = res_analysis.compute_median_all(results_df, 'rmse', self.task)
        
        # Export results
        if export:
            results_df.to_csv(f'{results_dir}/results_{self.seed}.csv', index=False)
            times_df.to_csv(f'{results_dir}/time_dim_{self.seed}.csv', index=False)
            final_res['model_median'].to_csv(f'{results_dir}/model_median_{self.seed}.csv', index=False)
            final_res['all_median'].to_csv(f'{results_dir}/all_median_{self.seed}.csv', index=False)
        
        # Report total time
        total_time = time.time() - total_start
        print(f"\nTotal experiment time: {total_time:.2f} seconds")
        
        return final_res

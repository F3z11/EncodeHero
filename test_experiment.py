"""Test script for the improved MLExperiment"""
import pandas as pd
from EncodeHero import MLExperiment

# Load mushrooms dataset
print("Loading mushrooms dataset...")
df = pd.read_csv("EncodeHero/mushrooms.csv")
print(f"Dataset shape: {df.shape}")
print(f"Target column values: {df['target'].unique()}")
print(f"Number of categorical columns: {len(df.select_dtypes(include=['object']).columns)}")

# Create experiment
print("\nCreating experiment...")
experiment = MLExperiment(data=df, target='target', task='binary', seed=42, dataset_name='mushrooms')

# Run experiment with caching enabled
print("\nRunning experiment (this may take a while)...")
results = experiment.run_experiment(export=True, dir='.', use_cache=True)

print("\n" + "="*80)
print("EXPERIMENT COMPLETED!")
print("="*80)
print(f"\nResults summary:")
print(f"- Total encoders tested: {len(results['time_dim'])}")
print(f"- Results shape: {results['res_raw'].shape}")
print(f"\nTop 5 encoders by median F1 score:")
print(results['all_median'].head())
print(f"\nExecution times (seconds):")
print(results['time_dim'].sort_values('time').head(10))

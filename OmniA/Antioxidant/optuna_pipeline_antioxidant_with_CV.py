import omnia
import pandas as pd
import os
from omnia.generics.pipeline.pipeline import Pipeline
from omnia.generics.pipeline.voting import VotingPipeline
from omnia.generics import Pipeline, pd, np
from omnia.generics.pipeline_optimization.pipeline_optimization import PipelineOptimization
from sklearn.model_selection import train_test_split

# I already give the data

# df_antioxidant_case_study=pd.read_csv("df_antioxidant_case_study.csv")
# x = df_antioxidant_case_study.drop(['label'], axis=1)
# y = df_antioxidant_case_study.loc[:, ['label']]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42,stratify=y)
# x_train.to_csv('x_train_antioxidant.csv', index=False)
# x_test.to_csv('x_test_antioxidant.csv', index=False)
# y_train.to_csv('y_train_antioxidant.csv', index=False)
# y_test.to_csv('y_test_antioxidant.csv', index=False)


x_train = pd.read_csv( 'x_train_antioxidant.csv')
x_test = pd.read_csv('x_test_antioxidant.csv')
y_train = pd.read_csv('y_train_antioxidant.csv')
y_test = pd.read_csv('y_test_antioxidant.csv')

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.15,stratify=y_train, random_state=42)


def run_optuna_pipeline_optimization(storage, study_name, direction, n_trials, x_train, y_train, x_val, y_val, metric, task_type, save_top_n, n_voting_pipelines, objective=CrossValidationObjective, trial_timeout,x_test,y_test):
    hpo = PipelineOptimization(storage=storage, study_name=study_name, direction=direction)
    hpo.optimize('omnia.proteins.light', n_trials=n_trials, x=x_train, y=y_train, x_val=x_val, y_val=y_val, metric=metric, task_type=task_type, save_top_n=save_top_n,
    n_voting_pipelines=n_voting_pipelines,objective=CrossValidationObjective, trial_timeout=trial_timeout, x_test, y_test, gc_after_trial=True)
    return hpo

# Parameters for the optimization
storage = 'sqlite:///antioxidant_with_cv_ssh.db'
study_name = 'antioxidant_with_cv'
direction = 'maximize'
n_trials = 100
metric = 'f1'
task_type = 'binary'
save_top_n = 50
n_voting_pipelines = 0
trial_timeout = 20000

# Run the optimization
hpo = run_optuna_pipeline_optimization(storage, study_name, direction, n_trials, x_train, y_train, x_val, y_val, metric, task_type, save_top_n, n_voting_pipelines, objective=CrossValidationObjective trial_timeout)
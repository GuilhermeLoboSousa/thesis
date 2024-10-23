from optuna.samplers import TPESampler
from omnia.generics import pd
from omnia.generics.pipeline_optimization.pipeline_optimization import PipelineOptimization
from sklearn.model_selection import train_test_split


def run_po():
    x_train = pd.read_csv( 'x_train_deepalgpro.csv')
    x_test = pd.read_csv('x_test_deepalgpro.csv')
    y_train = pd.read_csv('y_train_deepalgpro.csv')
    y_test = pd.read_csv('y_test_deepalgpro.csv')

    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.15,stratify=y_train, random_state=42)


    def run_optuna_pipeline_optimization(storage, study_name, direction, sampler, n_trials, x_train, y_train, x_val, y_val, metric, task_type, save_top_n, n_voting_pipelines, trial_timeout, x_test, y_test):
        hpo = PipelineOptimization(storage=storage, study_name=study_name, direction=direction, sampler=sampler, load_if_exists=True)
        hpo.optimize('omnia.proteins.light', n_trials=n_trials, x=x_train, y=y_train, x_val=x_val, y_val=y_val, metric=metric, task_type=task_type, save_top_n=save_top_n, n_voting_pipelines=n_voting_pipelines, trial_timeout=trial_timeout, x_test=x_test, y_test=y_test, gc_after_trial=True)
        return hpo

    # Parameters for the optimization
    storage = 'sqlite:///alergens_ssh.db'
    study_name = 'alergens'
    direction = 'maximize'
    sampler = TPESampler(seed=123,n_startup_trials=50)
    n_trials = 75
    metric = 'accuracy'
    task_type = 'binary'
    save_top_n = 50
    n_voting_pipelines = 3
    trial_timeout = 35000

    # Run the optimization
    hpo = run_optuna_pipeline_optimization(storage, study_name, direction, sampler, n_trials, x_train, y_train, x_val, y_val, metric, task_type, save_top_n, n_voting_pipelines, trial_timeout,x_test,y_test)

    trials_df = hpo.trials_dataframe()

    #savedataframe
    output_file = 'path.csv'
    trials_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    run_po()

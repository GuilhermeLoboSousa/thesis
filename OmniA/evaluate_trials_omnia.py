#An example of how to load the models and score

import os
import numpy as np
from omnia.generics import  Pipeline, pd
from omnia.proteins import ProteinStandardizer, Esm2Encoder

#put the coreect path for the test
x_test = pd.read_csv('x_test_deepalgpro.csv')
y_test = pd.read_csv('y_test_deepalgpro.csv')
#or
x_test = pd.read_csv('x_test_antioxidant.csv')
y_test = pd.read_csv('y_test_antioxidant.csv')

metrics_df = pd.DataFrame(columns=['trial','accuracy','balanced_accuracy','roc_auc','f1','recall','mcc'])
# test all trials under results_antioxidant_guilherme/antioxidant
for i, trial in enumerate(os.listdir('path')):
    # if pipeline-pkl in directory
    if 'pipeline.pkl' in os.listdir('path'+trial):
        
        # load pipeline
        pipeline = Pipeline.load('path'+trial)
        # score pipeline
        y_preds=pipeline.predict(x_test)
        
        try:
            metrics = pipeline.score(x_test, y_test, metrics=['accuracy','balanced_accuracy','roc_auc','f1','recall','mcc'])
        except:
            metrics = pipeline.score(x_test, y_test, metrics=['accuracy','balanced_accuracy','roc_auc','f1','recall','matthews_corrcoef'])
        # save metrics to dataframe
        metrics_df.loc[i] = [trial] + [metrics[metric] for metric in metrics]
        
metrics_df.to_csv('path.csv')
import os
import numpy as np
from omnia.generics import  Pipeline, pd
from omnia.proteins import ProteinStandardizer, Esm2Encoder

x_test = pd.read_csv('x_test_deepalgpro.csv')
y_test = pd.read_csv('y_test_deepalgpro.csv')

metrics_df = pd.DataFrame(columns=['trial','accuracy','balanced_accuracy','roc_auc','f1','recall','mcc'])
# test all trials under results_antioxidant_guilherme/antioxidant
for i, trial in enumerate(os.listdir('results_antioxidant_guilherme/antioxidant')):
    print(trial)
    # if pipeline-pkl in directory
    if 'pipeline.pkl' in os.listdir('results_antioxidant_guilherme/antioxidant/'+trial):
        
        # load pipeline
        pipeline = Pipeline.load('results_antioxidant_guilherme/antioxidant/'+trial)
        # score pipeline
        y_preds=pipeline.predict(x_test)
        y_preds=np.unique(y_preds)
        if np.any((y_preds !=0) and (y_preds !=1)):
            print(f"mais do que binary: {y_preds}")
        try:
            metrics = pipeline.score(x_test, y_test, metrics=['accuracy','balanced_accuracy','roc_auc','f1','recall','mcc'])
        except:
            metrics = pipeline.score(x_test, y_test, metrics=['accuracy','balanced_accuracy','roc_auc','f1','recall','matthews_corrcoef'])
        # save metrics to dataframe
        metrics_df.loc[i] = [trial] + [metrics[metric] for metric in metrics]
        
metrics_df.to_csv('results_antioxidant_guilherme/antioxidant/metrics.csv')
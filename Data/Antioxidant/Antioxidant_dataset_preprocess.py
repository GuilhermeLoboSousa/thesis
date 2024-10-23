# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC This notebbok is responsible for downloading the selected datasets, allowing us to improve our knowledge of them in terms of the distribution of sequence sizes, as well as the presence of possible duplicate sequences.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # imports

# COMMAND ----------

import csv
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import requests

# COMMAND ----------

# MAGIC %md
# MAGIC # Load data

# COMMAND ----------

df_antioxidant = pd.read_csv('df_antioxidant_case_study.csv')


# COMMAND ----------

# MAGIC %md
# MAGIC # shape

# COMMAND ----------

print("df_antioxidant",df_antioxidant.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC # Histogram

# COMMAND ----------

df_antioxidant['sequence_length'] = df_antioxidant['sequence'].apply(len)

bins = range(0, df_antioxidant['sequence_length'].max() + 100, 100)  # Intervals of 100

plt.figure(figsize=(10,6))
plt.hist(df_antioxidant['sequence_length'], bins=bins, edgecolor='black')
plt.title('Overview of sequence length distribution (Interval Size: 100 amino-acids)',fontsize=16)
plt.xlabel('Length of sequences',fontsize=16)
plt.ylabel('Number of sequences',fontsize=16)
plt.savefig('antioxidant_histogram.png', dpi=300)
plt.show()


# COMMAND ----------

# Criar o gráfico com histograma e curva KDE
plt.figure(figsize=(10,6))

# Adicionar histograma com intervalo de 50

# Adicionar a curva KDE com ajuste de largura da banda
sns.kdeplot(df_antioxidant['sequence_length'], color='red', linewidth=1, bw_adjust=0.25)

plt.xlim(left=0, right=1000)
plt.ylim(bottom=0) 

# Adicionar título e rótulos
plt.title('Distribution of Sequence Lengths with KDE Curve')
plt.xlabel('Length of Sequences')
plt.ylabel('Density')

# Exibir o gráfico
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Verify the presence of duplicate sequences

# COMMAND ----------

duplicates = df_antioxidant['sequence'].duplicated()

num_duplicates = duplicates.sum()
print(f"Number of duplicate sequences: {num_duplicates}")

if num_duplicates > 0:
    duplicate_sequences = df_antioxidant[df_antioxidant['sequence'].duplicated(keep=False)]
    print("Duplicate sequences:")
    print(duplicate_sequences)
else:
    print("No duplicate sequences found.")

# COMMAND ----------

df_antioxidant

# COMMAND ----------

# MAGIC %md
# MAGIC # Label values distribution

# COMMAND ----------


# Check the distribution of values in the 'label' column
label_counts = df_antioxidant['label'].value_counts()

# Create the pie chart
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired(range(len(label_counts))))

# Add the legend
plt.legend(wedges, label_counts.index, title="Labels", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),fontsize=14)

# Set the title in English
plt.title('Class Distribution (Label 0 vs Label 1)', fontsize=16)
plt.savefig('antioxidant_labels.png', dpi=300)

plt.show()

# COMMAND ----------

# Split the DataFrame by label
label_0_lengths = df_antioxidant[df_antioxidant['label'] == 0]['sequence_length']
label_1_lengths = df_antioxidant[df_antioxidant['label'] == 1]['sequence_length']

# Create histograms with customized colors
plt.hist(label_0_lengths, alpha=0.5, label='Label 0', bins=10)
plt.hist(label_1_lengths, alpha=0.5, label='Label 1', bins=10)

# Set labels and title
plt.xlabel('Sequence length',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.title('Length distribution of protein sequences by label',fontsize=15)

# Add legend
plt.legend(fontsize=15)
plt.savefig('antioxidant_histogram_labels.png', dpi=300)

# Display the plot
plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Exemplo de modelos
models = [ 'AodPred', 'AOP-SVM', 'PredAOP', 'Propythia','OmniA']

# Exemplo de métricas (substitua pelos seus valores reais)
metrics = ['Accuracy', 'Recall', 'MCC', 'F1 Score']
values = {
    'Accuracy': [74.79,94.2,93.18,91.10,69.0],
    'Recall': [74.48, 98.5, 96.77, 96.99, 63.2],
    'MCC': [36.8, 74.1, 71.2, 59.61, 68.50],
    'F1 Score': [45.2, 76.7, 74.9, 63.63, 71.6],
}
# Usar uma paleta de cores mais agradável (tab10)
cmap = cm.get_cmap('Dark2', len(models))
colors = cmap.colors

# Tamanho da figura
plt.figure(figsize=(12, 6))

# Número de barras por grupo
bar_width = 0.1  # Diminuir largura para dar mais espaço

# Espaço adicional entre grupos de métricas
spacing = 0.15

# Posições das barras no eixo x, considerando o espaçamento entre métricas
index = np.arange(len(metrics)) * (len(models) * bar_width + spacing)

# Criar barras para cada modelo
for i, model in enumerate(models):
    plt.bar(index + i * bar_width, [values[metric][i] for metric in metrics], 
            bar_width, label=model, color=colors[i])

# Configurações do gráfico
plt.xlabel('Metrics',fontsize=14)
plt.ylabel('Scores (%)',fontsize=14)
plt.title('Performance of different approaches on antioxidant dataset', fontsize=14)

# Definir os ticks do eixo y de 5 em 5
plt.yticks(np.arange(0, 101, 5),fontsize=14)

# Configuração do eixo x
plt.xticks(index + bar_width * 3, metrics,fontsize=14)

# Exibir legenda fora do gráfico
plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=13)
plt.savefig('antioxidant_global_sem_dpaop.png', dpi=300)

# Ajustar layout e mostrar gráfico
plt.tight_layout()
plt.show()


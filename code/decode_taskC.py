#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sys 

VAL_MODE=False


# In[3]:


val_filename = '../data/MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskC/TaskC-ValidationSet.csv'


# In[4]:


# Pass as CLI arg
# test_filename = '../data/MEDIQA-Chat-TestSets-March-15-2023/TaskC/taskC_testset4participants_inputNotes.csv'
test_filename = sys.argv[1]


# In[5]:


if VAL_MODE:
    test_filename = val_filename

print(f'Test input filename:{test_filename}\n')

# In[6]:


model_path = '../taskC-rouge_59_24'


# In[7]:


output_filename = './outputs/taskC_Cadence_run1.csv'


# In[8]:


temp_summarizer_output_filename = './outputs/temp/taskC-summarizer/summarizer-taskC_Cadence_run1.csv'


# In[9]:


# temp_classifier_output_filename = './outputs/temp/taskA-classifier/classifier-taskA_Cadence_run1.csv'


# In[10]:


# Run prediction
import os
os.system(f'python3 ./run_summarization-taskC.py      --model_name_or_path {model_path}      --do_predict      --output_dir  ./outputs/temp/taskC-summarizer      --test_file  {test_filename}      --overwrite_output_dir      --predict_with_generate      --text_column note      --save_strategy "epoch"      --generation_max_length 1024      --max_target_length 1024      --max_source_length 1024      --per_device_eval_batch_size 1')


# In[11]:


EXAMPLE_SEP = '\n###############END-PRED##################\n'


# In[12]:


summarizer_out_filename = './outputs/temp/taskC-summarizer/taskC_summarizer-generated_predictions.txt'


# In[13]:


summarizer_pred_out = ""

with open(summarizer_out_filename, "r") as f:
    for line in f.readlines():
        summarizer_pred_out+=line
        
summarized_preds = summarizer_pred_out.split(EXAMPLE_SEP)
print(f'Total summarized preds:{len(summarized_preds)}')


# In[14]:


import pandas as pd
def read_csv(filename):
    return pd.read_csv(filename)


# In[15]:


val_df = read_csv(val_filename)
val_df


# In[16]:


summarized_labels = list(val_df['dialogue'])
print(f'Total summarized labels:{len(summarized_labels)}')


# In[17]:


import evaluate 
rouge = evaluate.load('rouge')


# In[18]:


if VAL_MODE:
    result = rouge.compute(predictions=summarized_preds, references=summarized_labels, use_stemmer=True)
    print(result)


# In[19]:


test_df = read_csv(test_filename)
test_df


# In[20]:


id_column = 'TestID' if 'TestID' in test_df.columns else 'encounter_id'


# In[21]:


taskC_out_df = test_df[[id_column]]
taskC_out_df['SystemOutput'] = summarized_preds
taskC_out_df.rename(columns={id_column:'TestID'}, inplace=True)
taskC_out_df


# In[22]:


taskC_out_df.to_csv(output_filename, index=False)


# In[ ]:

print('Task C inference completed!')



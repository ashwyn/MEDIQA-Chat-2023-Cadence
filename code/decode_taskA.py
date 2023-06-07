#!/usr/bin/env python
# coding: utf-8

# In[2]:
import sys 
import os

os.system('conda run -n Cadence_tasks_venv python --version')

VAL_MODE=False


# In[3]:


val_filename = '../data/MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv'


# In[4]:


# Pass as CLI arg
# test_filename = '../data/MEDIQA-Chat-TestSets-March-15-2023/TaskA/taskA_testset4participants_inputConversations.csv'
test_filename = sys.argv[1]


# In[5]:


if VAL_MODE:
    test_filename = val_filename


print(f'Test input filename:{test_filename}\n')

# In[6]:


model_path = '../taskA-summ-rouge_51_21'


# In[7]:


output_filename = './outputs/taskA_Cadence_run1.csv'


# In[8]:


temp_summarizer_output_filename = './outputs/temp/taskA-summarizer/summarizer-taskA_Cadence_run1.csv'


# In[9]:


temp_classifier_output_filename = './outputs/temp/taskA-classifier/classifier-taskA_Cadence_run1.csv'


# In[10]:


# Run prediction
os.system(f'python3 ./run_summarization-taskA.py      --model_name_or_path {model_path}      --do_predict      --output_dir  ./outputs/temp/taskA-summarizer      --test_file  {test_filename}      --overwrite_output_dir      --predict_with_generate      --text_column dialogue      --save_strategy \"epoch\"      --generation_max_length 1024      --max_target_length 1024      --max_source_length 1024      --per_device_eval_batch_size 2')


# In[11]:


EXAMPLE_SEP = '\n###############END-PRED##################\n'


# In[12]:


summarizer_out_filename = './outputs/temp/taskA-summarizer/taskA_summarizer-generated_predictions.txt'


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


summarized_labels = list(val_df['section_text'])
print(f'Total summarized labels:{len(summarized_labels)}')


# In[17]:


import evaluate 
rouge = evaluate.load('rouge')


# In[18]:


if VAL_MODE:
    result = rouge.compute(predictions=summarized_preds, references=summarized_labels, use_stemmer=True)
    print(result)


# In[19]:


# Classification
idx2label = {0: 'OTHER_HISTORY', 1: 'DISPOSITION', 2: 'ALLERGY', 3: 'PROCEDURES', 4: 'EXAM', 5: 'IMAGING', 6: 'PASTMEDICALHX', 7: 'MEDICATIONS', 8: 'LABS', 9: 'GYNHX', 10: 'EDCOURSE', 11: 'ASSESSMENT', 12: 'GENHX', 13: 'ROS', 14: 'IMMUNIZATIONS', 15: 'FAM/SOCHX', 16: 'PLAN', 17: 'PASTSURGICAL', 18: 'CC', 19: 'DIAGNOSIS'}
label2idx = {'OTHER_HISTORY': 0, 'DISPOSITION': 1, 'ALLERGY': 2, 'PROCEDURES': 3, 'EXAM': 4, 'IMAGING': 5, 'PASTMEDICALHX': 6, 'MEDICATIONS': 7, 'LABS': 8, 'GYNHX': 9, 'EDCOURSE': 10, 'ASSESSMENT': 11, 'GENHX': 12, 'ROS': 13, 'IMMUNIZATIONS': 14, 'FAM/SOCHX': 15, 'PLAN': 16, 'PASTSURGICAL': 17, 'CC': 18, 'DIAGNOSIS': 19}


# In[20]:


test_df = read_csv(test_filename)
test_df


# In[21]:


classifier_model_path = '../taskA-class-acc_79-f1_79'

from transformers import pipeline
classifier = pipeline('text-classification', model=classifier_model_path)


test_dialogues = list(test_df['dialogue'])

classes_preds = classifier(test_dialogues )

classes_preds = [pred_result['label'] for pred_result in classes_preds]

classes_idx_preds = [label2idx[lab] for lab in classes_preds]


if VAL_MODE:
    classes_labels = list(test_df['section_header'])
    print(f'Total class labels:{len(classes_labels)}')
    
    classes_idx_labels = [label2idx[lab] for lab in classes_labels]

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    acc = accuracy.compute(predictions=classes_idx_preds, references=classes_idx_labels)
    f1_score = f1.compute(predictions=classes_idx_preds, references=classes_idx_labels, average='weighted')['f1']
    f1_score_non_wt = f1.compute(predictions=classes_idx_preds, references=classes_idx_labels, average=None)['f1']
    f1_score_macro = f1.compute(predictions=classes_idx_preds, references=classes_idx_labels, average='macro')['f1']

    print(f'Accuracy: {acc} | F1: {f1_score} | F1 (non-weighted): {f1_score_non_wt} | F1 (macro): {f1_score_macro}')


# In[22]:


id_column = 'TestID' if 'TestID' in test_df.columns else 'ID'


# In[23]:


taskA_out_df = test_df[[id_column]]
taskA_out_df['SystemOutput1'] = classes_preds
taskA_out_df['SystemOutput2'] = summarized_preds
taskA_out_df.rename(columns={id_column:'TestID'}, inplace=True)
taskA_out_df


# In[24]:


taskA_out_df.to_csv(output_filename, index=False)


# In[ ]:


print('Task A inference completed!')


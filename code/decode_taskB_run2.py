#!/usr/bin/env python
# coding: utf-8

# In[2]:
import sys

VAL_MODE=False


# In[3]:


val_filename = '../data/MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskB/TaskB-ValidationSet.csv'


# In[4]:


# Pass as CLI arg
#test_filename = '../data/MEDIQA-Chat-TestSets-March-15-2023/TaskB/taskB_testset4participants_inputConversations.csv'
test_filename = sys.argv[1]

# In[5]:


if VAL_MODE:
    test_filename = val_filename


# In[6]:


model_path = '../taskB-rouge_54_26'


# In[7]:


second_pass_model_path = '../taskB-rouge_58_28_2nd_pass'


# In[8]:


output_filename = './outputs/taskB_Cadence_run2.csv'


# In[9]:


temp_summarizer_output_filename = './outputs/temp/taskB-summarizer/summarizer-taskB_Cadence_run1.csv'


# In[10]:


# temp_classifier_output_filename = './outputs/temp/taskA-classifier/classifier-taskA_Cadence_run1.csv'


# In[11]:


# Run prediction
import os

os.system(f'python3 ./run_summarization-taskB.py      --model_name_or_path {model_path}      --do_predict      --output_dir  ./outputs/temp/taskB-summarizer      --test_file  {test_filename}      --overwrite_output_dir      --predict_with_generate      --text_column dialogue      --save_strategy "epoch"      --generation_max_length 1024      --max_target_length 1024      --max_source_length 1024      --per_device_eval_batch_size 1')


# In[12]:


EXAMPLE_SEP = '\n###############END-PRED##################\n'


# In[13]:


summarizer_out_filename = './outputs/temp/taskB-summarizer/taskB_summarizer-generated_predictions.txt'


# In[14]:


summarizer_pred_out = ""

with open(summarizer_out_filename, "r") as f:
    for line in f.readlines():
        summarizer_pred_out+=line
        
summarized_preds = summarizer_pred_out.split(EXAMPLE_SEP)
print(f'Total summarized preds:{len(summarized_preds)}')


# In[15]:


import pandas as pd
def read_csv(filename):
    return pd.read_csv(filename)


# In[16]:


val_df = read_csv(val_filename)
val_df


# In[17]:


summarized_labels = list(val_df['note'])
print(f'Total summarized labels:{len(summarized_labels)}')


# In[18]:


import evaluate 
rouge = evaluate.load('rouge')


# In[19]:


if VAL_MODE:
    result = rouge.compute(predictions=summarized_preds, references=summarized_labels, use_stemmer=True)
    print(result)


# In[20]:


test_df = read_csv(test_filename)
test_df


# In[21]:


id_column = 'TestID' if 'TestID' in test_df.columns else 'encounter_id'


# In[22]:


taskB_out_df = test_df[[id_column]]
taskB_out_df['SystemOutput'] = summarized_preds
taskB_out_df.rename(columns={id_column:'TestID'}, inplace=True)
taskB_out_df


# In[23]:


taskB_out_df.to_csv(output_filename, index=False)


# In[24]:


# 2nd pass analysis


# In[25]:


import pandas as pd
def read_from_csv(filename):
    return pd.read_csv(filename)


# In[26]:


# Prep Input (first half summary + second half dialogue)
test_df = read_from_csv(test_filename)
test_df


# In[27]:


first_half_summary_df = read_from_csv(output_filename)
first_half_summary_df


# In[28]:


FIRST_HALF_SUMM_TAG = 'Context:\n'
SECOND_HALF_DIAL_TAG = 'Dialogue:\n'


# In[29]:


test_dialogues = list(test_df['dialogue'])
first_half_summaries = list(first_half_summary_df['SystemOutput'])
test_second_half_dialogues = [dial[int(len(dial)/2):] for dial in test_dialogues]
test_first_half_summary_second_half_dialogues = [FIRST_HALF_SUMM_TAG + first_half_summaries[idx] + SECOND_HALF_DIAL_TAG + test_second_half_dialogues[idx] for idx in range(len(test_second_half_dialogues))]


# In[30]:


test_df['first_half_summary_second_half_dialogue'] = test_first_half_summary_second_half_dialogues
test_df


# In[31]:


print(test_first_half_summary_second_half_dialogues[10])


# In[32]:


test_df.rename(columns={id_column:'TestID'}, inplace=True)
test_df


# In[33]:


test_df['dataset'] = ['dummy']*len(list(test_df['TestID']))
test_df['encounter_id'] = ['dummy']*len(list(test_df['TestID']))
test_df['TestID'] = [str(tid) for tid in list(test_df['TestID'])]
test_df = test_df[['dataset','encounter_id','dialogue','first_half_summary_second_half_dialogue']]
test_df


# In[34]:


second_pass_test_filename = f'{test_filename.split(".csv")[0]}_secondpass.csv'
test_df.to_csv(second_pass_test_filename)


# In[35]:


# Train on 2nd pass summarization
# Run prediction

os.system(f'python3 ./run_summarization-taskB.py      --model_name_or_path {second_pass_model_path}      --do_predict      --output_dir  ./outputs/temp/taskB-summarizer      --test_file  {second_pass_test_filename}      --overwrite_output_dir      --predict_with_generate      --text_column first_half_summary_second_half_dialogue      --save_strategy "epoch"      --generation_max_length 1024      --max_target_length 1024      --max_source_length 1024      --per_device_eval_batch_size 1      --fp16')


# In[ ]:





# In[36]:


EXAMPLE_SEP = '\n###############END-PRED##################\n'


# In[37]:


summarizer_out_filename = './outputs/temp/taskB-summarizer/taskB_summarizer-generated_predictions.txt'


# In[38]:


summarizer_pred_out = ""

with open(summarizer_out_filename, "r") as f:
    for line in f.readlines():
        summarizer_pred_out+=line
        
summarized_preds = summarizer_pred_out.split(EXAMPLE_SEP)
print(f'Total summarized preds:{len(summarized_preds)}')


# In[39]:


import pandas as pd
def read_csv(filename):
    return pd.read_csv(filename)


# In[40]:


val_df = read_csv(val_filename)
val_df


# In[41]:


summarized_labels = list(val_df['note'])
print(f'Total summarized labels:{len(summarized_labels)}')


# In[42]:


import evaluate 
rouge = evaluate.load('rouge')


# In[43]:


if VAL_MODE:
    result = rouge.compute(predictions=summarized_preds, references=summarized_labels, use_stemmer=True)
    print(result)


# In[44]:


test_df = read_csv(test_filename)
test_df


# In[45]:


id_column = 'TestID' if 'TestID' in test_df.columns else 'encounter_id'


# In[46]:


taskB_out_df = test_df[[id_column]]
taskB_out_df['SystemOutput'] = summarized_preds
taskB_out_df.rename(columns={id_column:'TestID'}, inplace=True)
taskB_out_df


# In[47]:


taskB_out_df.to_csv(output_filename, index=False)


# In[ ]:

print('TaskB - run2 inference completed!')



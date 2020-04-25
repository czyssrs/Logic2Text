# Logic2Text

## Data
In the dataset folder, we have the full dataset (all_data.json), and the train test split (train.json, valid.json, test.json). 
Each example is in a dictionary of the following format:
```
    "topic": table caption,
    "wiki": link to the wikipedia page,
    "action": logic type,
    "sent": description,
    "annotation": raw annotation data from the mturk,
    "logic": logic form,
    "logic_str": linearized logic form,
    "nid": number of fuction nodes,
    "table_header": table header, in list format
    "table_cont": table content, list of lists
  
```

## Logic form execution
In the execution folder, run
```
python execute.py
```
It will execute all the logic forms in all_data.json. All the function definitions are in APIs.py

This site is under construction, and we will release other codes in the future.

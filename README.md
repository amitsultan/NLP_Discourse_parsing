# Discourse Parsing Project - NLP Course 2023
## Submitters
### _Amit Sultan_ - _ID: 205975444_
### _Efrat Cohen_ - _ID: 207783150_
### _Liat Cohen_ - _ID: 205595283_
### _Liel Serfaty_ - _ID: 312350622_


### Background
In recent years discourse parsing in Natural Language Processing (NLP) has been a topic of interest,
given its potential to extract important information and relationships between texts. The growth of online argumentation 
platforms has given researchersthe chance to create computational approaches for a broader-scale investigation of the key 
elements of persuasiveness, such as language use,audience traits (such as demographics, preconceived notions) and so-cial 
interactions. The recognition that discourse units (at their most basic, clauses) are connected to one another in principled 
ways, and that the conjunction of two units produces a combined meaning greater than each unit’s alone, forms the basis of 
discourse structure.

### Code overview
A brief overview of our code files and run-process:

- _*BertController*_ - Bert Controller is responsible for handling transformers.bert this includes preparing the model 
   for various input sizes as well as output sizes. SUPPORT CUDA / CPU. It is also responsible of loading pre-trained models
   that we previously trained such as C3 and FeedBack Datasets.
- _*DataLoaderGenetric*_ - Data Loader Generic is responsible for loading the different dataset and uniformly create text and 
  labels suited for bert model. It helps load the various dataset and generalize the text\labels tuple as for some dataset
  need additional actions (Discourse -> creating the tress).
- _*Helper*_ - General methods used to calculate metrics and it contains constant with the label names and column names.


## Example NoteBook
We created a single NoteBook for this project, the noetbook includes all the segments starting from imports, loading datasets and
training each model on a different dataset. We will explain each segment in short and the full code can be found at the notebook itself.

- **Imports**
- **C3 Training and saving model** - The following code contains the loading of the C3 dataset using our generic DataLoader and then finetuning
 a bert model to fit the data, evaluating our performance on a validation set.
- **FeedBack Training and saving model** - The following code contains the loading of the FeedBack dataset using our generic DataLoader and then finetuning
 a bert model to fit the data, evaluating our performance on a validation set.
- **Loading of discourse dataset and generic function for pre-trained models loading and testing** - Now we wish to test each of the model trained in the
  previous parts and fine-tune them to our discourse task, we load the pre-trained model and continue to train and evaluation each dataset.
- **Discourse trained on feedback** - Pre-trained model on a feedback dataset used to train the discourse parsing task.
- **Discourse trained on C3** - Pre-trained model on a C3 dataset used to train the discourse parsing task.
- **Discourse trained on each label** - The last part is similar to the above but we train the discourse task on each label separately.


git link: https://github.com/amitsultan/NLP_Discourse_parsing
This data bundle is used with our ICWSM 2021 paper Discourse Parsing for Contentious, Non-Convergent Online Discussions. URL: https://ojs.aaai.org/index.php/ICWSM/article/view/18109


It contains three files:
1. This README file
2. annotated_trees_101.csv: An annotated dataset of 101 conversation trees
3. CDP_coding_scheme.pdf:  Tagset specifications and examples


Bibtex:
@article{Zakharov2021, 
title={Discourse Parsing for Contentious, Non-Convergent Online Discussions}, volume={15}, url={https://ojs.aaai.org/index.php/ICWSM/article/view/18109},
journal={Proceedings of the International AAAI Conference on Web and Social Media}, author={Zakharov, Stepan and Hadar, Omri and Hakak, Tovit and Grossman, Dina and Ben-David Kolikant, Yifat and Tsur, Oren}, 
year={2021}, 
month={May}, 
pages={853-864},
}




Loading the data  within a Python environment is straightforward:
import pandas as pd
df = pd.read_csv('annotated_trees_101.csv')


Loading the file results in a dataframe with 10,559 rows+header (each row representing a node, or post) with 7 major columns + 31 tag columns (total 38 columns): 
1. rowid (from 0 to 10,559)
2. node_id (as fetched from reddit)
3. tree_id (as fetched from reddit)
4. timestamp (unix format, as fetched from reddit)
5. author name of the comment
6. text of the comment
7. parent: the rowid of the parent of the current node (-1 for root, 101 total roots)
8. And 31 columns for tags. (values: 0 & 1). The tags after consolidation, just as we have used them in our ML experiments.
import pandas as pd


COLUMN_CSV = ['Subject', 'S6_Diff']
data_path  = '../resources/subjects_labels/subject_S6Diff.xls'

# COLUMN_CSV = ['Subject', 'S_Min']
# data_path = '../resources/subjects_labels/subject_Smin.xls'

def load_dict(path):

   file_obj = pd.read_excel(path, names=COLUMN_CSV, header=0)
   subjects = file_obj.iloc[:, 0]
   labels = file_obj.iloc[:, 1]
   sub_to_label = {}
   cnt = 0
   for i in range(0, len(subjects)):
       sub_to_label[subjects[i]] = labels[i]
   return sub_to_label, cnt


sub_to_label = load_dict(data_path)
print (sub_to_label)
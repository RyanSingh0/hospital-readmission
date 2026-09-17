# Data

Neither dataset is redistributed here, and both are publicly available.

## Diabetes 130-US Hospitals (primary cohort)

Download `dataset_diabetes.zip` from the UCI Machine Learning Repository, then place
`diabetic_data.csv` and `IDS_mapping.csv` in this directory.

    https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

The file carries 101,766 encounters across 50 columns, covering 130 US hospitals between
1999 and 2008.

## Synthetic 30k readmission set (secondary cohort)

Download `hospital_readmissions_30k.csv` from Kaggle and place it here.

    https://www.kaggle.com/datasets/siddharth0935/hospital-readmission-predictionsynthetic-dataset

This cohort is used only in experiment 01, which establishes that the file was generated
from a two-branch rule, and that no model can exceed an AUC of roughly 0.581 on it.

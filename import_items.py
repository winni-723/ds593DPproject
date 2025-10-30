
import pandas as pd
df= pd.read_csv("RateMyProfessor_Sample_data.csv")
df['professor_name'] = df['professor_name'].astype(str).str.split().str.join(' ')
selected = df[['professor_name','school_name','department_name','star_rating','name_not_onlines','student_difficult','would_take_agains','help_useful','comments']]
#print(df['professor_name'].head(10).tolist())
print(df['help_useful'].unique())
#selected.to_csv('data.csv', index=False)
#print(len(selected.columns))

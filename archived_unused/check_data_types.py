import pandas as pd

df = pd.read_csv('data/ml_training_data_1year.csv')

print('Non-numeric columns:')
non_num = df.select_dtypes(exclude=['number']).columns.tolist()
print(non_num)

print('\nSample values from non-numeric columns:')
for col in non_num[:10]:
    print(f'{col}: {df[col].unique()[:5].tolist()}')

print('\n\nNumeric columns:', len(df.select_dtypes(include=['number']).columns))
print('Non-numeric columns:', len(non_num))

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
print(os.getcwd())

df = pd.read_csv('data/train.csv')
df.dropna(inplace=True)
print("\nTraining Data Info:")
print(df.info())

name_list = df['Name'].tolist()
unique_names = len(np.unique(df['Name'].tolist()))
print(f"Unique Names: {unique_names}")
unique_sex = len(np.unique(df['Sex'].tolist()))
print(f"Unique Sex: {unique_sex}")
unique_tickets = len(np.unique(df['Ticket'].tolist()))
print(f"Unique Tickets: {unique_tickets}")
unique_embarked = len(np.unique(df['Embarked'].tolist()))
print(f"Unique Embarked: {unique_embarked}")

split_value = df['Ticket'].str.split(' ')
df['TicketNumber'] = split_value.str[-1]
df['TicketPrefix'] = split_value.str[:-1].apply(lambda x: ' '.join(x) if len(x) > 0 else '')
print("\nTicket Number and Prefix Info:")
print(df[['Ticket', 'TicketNumber', 'TicketPrefix']].head())


res = {}
for i in range(len(df['TicketPrefix'])):
    print(f"{i}: {df.iloc[i]['TicketPrefix']}")
    res[df.iloc[i]['TicketPrefix']] = res.get(df.iloc[i]['TicketPrefix'], 0) + 1
print(res)

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

df['HasPrefix'] = df['TicketPrefix'].apply(lambda x: 0 if x == '' else 1)
df['TicketLength'] = df['TicketNumber'].astype(str).apply(len)
df['TicketIsLine'] = df['TicketNumber'].apply(lambda x: 1 if str(x).upper() == 'LINE' else 0)

df.to_csv('data/train_preprocessed.csv', index=False)

plt.figure(figsize=(10, 8))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.title('Correlation Heatmap')
plt.tight_layout()
Path('figures').mkdir(parents=True, exist_ok=True)
plt.savefig('figures/correlation_heatmap.png', bbox_inches='tight')
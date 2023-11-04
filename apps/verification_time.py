import pandas as pd

df = pd.read_csv('verify2s.csv')

# print('pair number =', len(df))

com通过 = df[df['com']=='proved']
com没通过 = df[df['com']!='proved']

sem通过 = df[df['sem']=='proved']
sem没通过 = df[df['sem']!='proved']

print('total com', df['com_time'].sum())
print('total sem', df['sem_time'].sum())

print('passed check time', com通过['com_time'].sum() + sem通过['sem_time'].sum())
print('failed check time', com没通过['com_time'].sum() + sem没通过['sem_time'].sum())


com1 = len(com通过)
print('#com pass =', com1)
print('#com fail =', len(com没通过))
com2 = com通过['com_time'].mean()
print('avg passed com = {:.3}'.format( com2))
com3 = df[df['com']!='proved']['com_time'].mean()
print('avg failed com = {:.3}'.format( com3))
print('{} & {:.3} & {:.3}'.format( com1, com2, com3))


sem通过 = df[df['sem']=='proved']
sem1 = len(sem通过)
print('#sem pass =', len(sem通过))
print('#sem fail =', len(sem没通过))
sem2 = sem通过['sem_time'].mean()
print('avg passed sem = {:.3}'.format( sem2))
sem3 = ( df[df['sem']!='proved']['sem_time'].mean())
print('avg failed sem = {:.3}'.format(sem3))
print('{} & {:.3} & {:.3}'.format( sem1, sem2, sem3))



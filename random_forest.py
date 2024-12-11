from methods import *

s=0
while s<500:
  generated_dataset = balanced_generate_dataset(50, 50)
  generated_dataset.reset_index(drop=True, inplace=True)
  s=len(generated_dataset)
random_forest_model = Random_Forest(trees_count=200, dataset=generated_dataset, phi=0.01055)

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(random_forest_model, f)
conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY,
        model BLOB
    )
''')

with open('random_forest_model.pkl', 'rb') as f:
    model_blob = f.read()
    cursor.execute('INSERT INTO models (model) VALUES (?)', (model_blob,))

conn.commit()
conn.close()


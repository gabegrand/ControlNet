import json
from sklearn.model_selection import train_test_split

data = []
with open('data/fill50k/prompt.json', 'rt') as f:
    for line in f:
        data.append(json.loads(line))

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

json.dump(train_data, open('data/fill50k/train.json', 'w'))
json.dump(val_data, open('data/fill50k/val.json', 'w'))
json.dump(test_data, open('data/fill50k/test.json', 'w'))

for split, data_split in [('train', train_data), ('val', val_data), ('test', test_data)]:
    with open(f'data/{split}.json', 'w') as f:
        for d in data_split:
            json.dump(d, f)
            f.write('\n')

import matplotlib.pyplot as plt
import yaml

with open('./data/training_data.yml') as file:
    data = yaml.safe_load(file)

print(data)

w = data['train_dataset'][0]['total']['subgroups'][0]['gender']
y = w.values()
x = w.keys()
#[0]['subgroups'][0].values()
print(y)
print(x)

# dict_keys is iterable but not indexable, so below gives way to index
# ATA - see favourites for recommended way
#for key in y:
#    print(key)
#    z= data['train_dataset'][0][key]
#    print(z)

fig, ax = plt.subplots()

#bar_labels = ['red', 'blue']
#bar_colors = ['tab:red', 'tab:blue']

ax.bar(x, y)

ax.set_ylabel('Patients')
ax.set_title('Patient demographics')
ax.set_yticklabels(tick_labels.astype(int))
#ax.legend(title='Fruit color')

plt.show()
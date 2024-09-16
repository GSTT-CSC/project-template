import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw

# read data from yaml file
with open('./data/training_data.yml') as file:
    data = yaml.safe_load(file)
# needed so that the graphs can be given device specific names in website repo
with open('./data/device.yml') as file2:
    device = yaml.safe_load(file2)
name = device['name']

# plot sex graph
# would be better to have all plotting in for loop, when doing this note dict_keys is iterable but not indexable
try:
  w = data['train_dataset'][0]['total']['subgroups'][0]['gender']
  y = w.values()
  x = w.keys()

  fig, ax = plt.subplots()
  fig.set_size_inches(5.6, 4.8)
  bars = ax.bar(x, y, color=(0.06, 0.38, 0.94, 0.7))

# formatting graph
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_color('#DDDDDD')

  ax.tick_params(bottom=False, left=False)

  ax.set_axisbelow(True)
  ax.yaxis.grid(False)
  ax.xaxis.grid(False)

# add text above bars, same color as bars
  bar_color = bars[0].get_facecolor()

  for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        # may want to add a small number to this param to give a small gap between text and bar
        bar.get_height(),
        bar.get_height(),
        horizontalalignment='center',
        color=bar_color,
        weight='bold'
    )

  ax.set_xlabel('Sex', labelpad=15, color='#333333')
  ax.set_ylabel('Number of Patients', labelpad=15, color='#333333')
  ax.set_title('Patients in Training Dataset by Sex', pad=15, color='#333333',
             weight='bold')
  fig.tight_layout()

# save fig and clear for next one
  plt.savefig("demo_graphs/"+name+"_sex_graph.png")
  plt.clf()

except:
  img = Image.new(mode="RGBA", size=(480,480), color='lightgrey')
  draw = ImageDraw.Draw(img)
  text = "No Image Available"
  draw.text((10,10), text, fill=(0,0,0))
  img.save("demo_graphs/"+name+"_sex_graph.png")


# plot ethnicity graph
try:
  w = data['train_dataset'][0]['total']['subgroups'][1]['ethnicity']
  y = w.values()
  x = w.keys()

  fig, ax = plt.subplots()
  fig.set_size_inches(8, 4.8)
  bars = ax.bar(x, y, color=(0.06, 0.38, 0.94, 0.7))

  # formatting graph
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_color('#DDDDDD')

  ax.tick_params(bottom=False, left=False)

  ax.set_axisbelow(True)
  ax.yaxis.grid(False)
  ax.xaxis.grid(False)

  # add text above bars, same color as bars
  bar_color = bars[0].get_facecolor()

  for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        bar.get_height(),
        horizontalalignment='center',
        color=bar_color,
        weight='bold'
    )

  ax.set_xlabel('Ethnicity', labelpad=15, color='#333333')
  ax.set_ylabel('Number of Patients', labelpad=15, color='#333333')
  ax.set_title('Patients in Training Dataset by Ethnicity', pad=15, color='#333333',
             weight='bold')
  fig.tight_layout()

  # save fig and clear for next one
  plt.savefig("demo_graphs/"+name+"_ethn_graph.png")
  plt.clf()

except:
  img = Image.new(mode="RGBA", size=(480,480), color='lightgrey')
  draw = ImageDraw.Draw(img)
  text = "No Image Available"
  draw.text((10,10), text, fill=(0,0,0))
  img.save("demo_graphs/"+name+"_ethn_graph.png")


# plot age graph
try:
  w = data['train_dataset'][0]['total']['subgroups'][2]['Age']
  y = w.values()
  x = w.keys()

  fig, ax = plt.subplots()
  fig.set_size_inches(11, 4.8)
  bars = ax.bar(x, y, color=(0.06, 0.38, 0.94, 0.7))

  # formatting graph
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_color('#DDDDDD')

  ax.tick_params(bottom=False, left=False)

  ax.set_axisbelow(True)
  ax.yaxis.grid(False)
  ax.xaxis.grid(False)

  # add text above bars, same color as bars
  bar_color = bars[0].get_facecolor()

  for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        bar.get_height(),
        horizontalalignment='center',
        color=bar_color,
        weight='bold'
    )

  ax.set_xlabel('Age', labelpad=15, color='#333333')
  ax.set_ylabel('Number of Patients', labelpad=15, color='#333333')
  ax.set_title('Patients in Training Dataset by Age', pad=15, color='#333333',
             weight='bold')
  fig.tight_layout()

  # save fig
  plt.savefig("demo_graphs/"+name+"_age_graph.png")

except:
  img = Image.new(mode="RGBA", size=(480,480), color='lightgrey')
  draw = ImageDraw.Draw(img)
  text = "No Image Available"
  draw.text((10,10), text, fill=(0,0,0))
  img.save("demo_graphs/"+name+"_age_graph.png")


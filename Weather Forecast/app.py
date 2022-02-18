from xmlrpc.client import Boolean
import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 5)
    self.fc2 = nn.Linear(5, 3)
    self.fc3 = nn.Linear(3, 1)
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from flask import Flask, request, render_template

app = Flask(__name__)
net = torch.load('model.pth')

@app.route('/')
def wforecaste():
    return render_template('wforecast.html',template_folder='flask')

@app.route('/predict', methods=['GET','POST'])
def predict():
  rainfall = request.form['rainfall']
  rainfall = float(rainfall)
  humidity = request.form['humidity']
  humidity = float(humidity)
  rain_today = request.form['rain_today']
  rain_today = Boolean(rain_today)
  pressure = request.form['pressure']
  pressure =float(pressure)
 
  t = torch.as_tensor([rainfall, humidity, rain_today, pressure]) \
      .float() \
      .to(device)
  output = net(t)
  b= output.ge(0.5).item()
  if b==True:
    return render_template('wforecast.html', prediction = "Tomorrow will be a Rainy day")
  else:
    return render_template('wforecast.html', prediction = "Tomorrow will be a Sunny day")

if __name__ == '__main__':
   app.run(debug = True)


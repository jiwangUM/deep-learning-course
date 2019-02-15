from logistic import *
from solver import *
import pickle

with open('data.pkl','rb') as f:
    raw_data = pickle.load(f,encoding='latin1')
    
X_raw, y_raw = raw_data
N = X_raw.shape[0]
data_split_percent = [0.6, 0.2, 0.2]
N_train = int(data_split_percent[0] * N)
#Bug3: N_val means the index of end of the validation set, not number of val data
#       Otherwise'X_val': X_raw[N_train:N_val] --> X_val is empty
N_val   = N_train + int(data_split_percent[1] * N)

data = {
        'X_train': X_raw[:N_train],
        'y_train': y_raw[:N_train],
        'X_val': X_raw[N_train:N_val],
        'y_val': y_raw[N_train:N_val],
        }

model = LogisticClassifier(input_dim=X_raw.shape[1], hidden_dim=10, weight_scale=1e-3, reg=0.001)


solver = Solver(model, data,
              #update_rule='rmsprop',
              #update_rule='adam',
              update_rule='sgd_momentum',
              #update_rule='sgd',
              optim_config={
                'learning_rate': 1e-2,
              },
              lr_decay=1,
              num_epochs=2500, batch_size=100,
              print_every=100)

solver.train()

X_test = X_raw[N_val:]
y_test = y_raw[N_val:]

test_acc = solver.check_accuracy(X_test, y_test)
print(test_acc) 
#(Iteration 14901 / 15000) loss: 0.187986
#(Epoch 2500 / 2500) train acc: 0.971667; val_acc: 0.950000
#test_acc = 0.93
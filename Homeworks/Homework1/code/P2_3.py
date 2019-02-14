from logistic import *
from solver import *
import pickle

with open('data.pkl','rb') as f:
    raw_data = pickle.load(f,encoding='latin1')
    
X_raw, y_raw = raw_data
N = X_raw.shape[0]
data_split_percent = [0.6, 0.2, 0.2]
N_train = int(data_split_percent[0] * N)
N_val   = int(data_split_percent[1] * N)
N_test  = N - N_train - N_val

data = {
        'X_train': X_raw[:N_train],
        'y_train': y_raw[:N_train],
        'X_val': X_raw[N_train:N_val],
        'y_val': y_raw[N_train:N_val],
        }

model = LogisticClassifier(input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0)


solver = Solver(model, data,
              update_rule='sgd',
              optim_config={
                'learning_rate': 1e-3,
              },
              lr_decay=0.95,
              num_epochs=10, batch_size=100,
              print_every=100)

solver.train()

X_test = X_raw[N_val:N]
y_test = y_raw[N_val:N]

test_acc = solver.check_accuracy(X_test, y_test)
print(test_acc)
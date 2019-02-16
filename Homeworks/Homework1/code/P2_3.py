from logistic import *
from solver import *
import pickle
import matplotlib.pyplot as plt

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

model = LogisticClassifier(input_dim=X_raw.shape[1], hidden_dim=None, weight_scale=1e-2, reg=0.0)


solver = Solver(model, data,
              update_rule='rmsprop',
              optim_config={
                'learning_rate': 1e-2,
              },
              lr_decay=1,
              num_epochs=3000, batch_size=100,
              print_every=100)

solver.train()

X_test = X_raw[N_val:]
y_test = y_raw[N_val:]

test_acc = solver.check_accuracy(X_test, y_test)
print(test_acc)

plt.plot(solver.val_acc_history, label="val_acc")
plt.plot(solver.train_acc_history, label="train_acc")
plt.legend(loc="lower right")
#(Iteration 17901 / 18000) loss: 0.117429
#(Epoch 3000 / 3000) train acc: 0.966667; val_acc: 0.945000
#test_acc = 0.935
from softmax import *
from solver import *
import pickle
import matplotlib.pyplot as plt


with open('mnist.pkl','rb') as f:
    raw_data = pickle.load(f,encoding='latin1')
    
train_set, val_set, test_set = raw_data

X_train, y_train = train_set
X_val, y_val     = val_set
X_test, y_test   = test_set

N_train = X_train.shape[0]
N_val   = X_val.shape[0]
N_test  = X_test.shape[0]

data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        }

model = SoftmaxClassifier(input_dim=X_train.shape[1], hidden_dim=15, weight_scale=1e-2, reg=0.001)


solver = Solver(model, data,
              #update_rule='rmsprop',
              update_rule='sgd_momentum',
              optim_config={
                'learning_rate': 1e-2,
              },
              lr_decay=0.99995,
              num_epochs=150, batch_size=100,
              print_every=100)

solver.train()

test_acc = solver.check_accuracy(X_test, y_test)
print(test_acc)

plt.plot(solver.val_acc_history, label="val_acc")
plt.plot(solver.train_acc_history, label="train_acc")
plt.legend(loc="lower right")

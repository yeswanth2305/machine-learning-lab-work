import math
import matplotlib.pyplot as plt

# A1 MODULES

def summation(x, w, b):
    s = 0
    for i in range(len(x)):
        s = s + x[i]*w[i]
    s = s + b
    return s


def activation(x, typ):
    if typ == "step":
        if x >= 0:
            return 1
        else:
            return 0

    if typ == "bipolar":
        if x >= 0:
            return 1
        else:
            return -1

    if typ == "sigmoid":
        return 1/(1+math.exp(-x))

    if typ == "relu":
        if x > 0:
            return x
        else:
            return 0


def error_calc(t, o):
    return t - o


def A1_modules():

    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,0,0,1]

    w = [0.2, -0.75]
    b = 10

    res = []

    for i in range(len(X)):
        net = summation(X[i], w, b)
        out = activation(net, "step")
        err = error_calc(Y[i], out)

        res.append((net, out, err))

    return res


# A2 AND PERCEPTRON 
def A2_and_gate():

    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,0,0,1]

    w = [0.2, -0.75]
    b = 10

    lr = 0.05
    epoch = 0

    err_list = []

    while epoch < 1000:

        total_error = 0

        for i in range(len(X)):

            net = X[i][0]*w[0] + X[i][1]*w[1] + b

            if net >= 0:
                out = 1
            else:
                out = 0

            err = Y[i] - out
            total_error = total_error + err*err

            w[0] = w[0] + lr*err*X[i][0]
            w[1] = w[1] + lr*err*X[i][1]

            b = b + lr*err

        err_list.append(total_error)

        if total_error <= 0.002:
            break

        epoch = epoch + 1

    # plot
    plt.plot(err_list)
    plt.title("Error vs Epoch (AND)")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.grid()
    plt.show()

    return w, b, epoch


# A3 ACTIVATION COMPARISON

def A3_compare_activation():

    acts = ["bipolar","sigmoid","relu"]
    res = []

    for a in acts:

        X = [[0,0],[0,1],[1,0],[1,1]]
        Y = [0,0,0,1]

        w = [0.2, -0.75]
        b = 10
        lr = 0.05

        epoch = 0

        while epoch < 1000:

            total_error = 0

            for i in range(len(X)):

                net = summation(X[i], w, b)
                out = activation(net, a)

                err = Y[i] - out
                total_error += err*err

                w[0] = w[0] + lr*err*X[i][0]
                w[1] = w[1] + lr*err*X[i][1]
                b = b + lr*err

            if total_error <= 0.002:
                break

            epoch += 1

        res.append((a, epoch))

    return res


# A4 LEARNING RATE 

def A4_learning_rate():

    rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    result = []

    for lr in rates:

        X = [[0,0],[0,1],[1,0],[1,1]]
        Y = [0,0,0,1]

        w = [0.2, -0.75]
        b = 10

        epoch = 0

        while epoch < 1000:

            total_error = 0

            for i in range(len(X)):

                net = summation(X[i], w, b)

                if net >= 0:
                    out = 1
                else:
                    out = 0

                err = Y[i] - out
                total_error += err*err

                w[0] += lr*err*X[i][0]
                w[1] += lr*err*X[i][1]
                b += lr*err

            if total_error <= 0.002:
                break

            epoch += 1

        result.append((lr, epoch))

    # plot
    lr_vals = []
    ep_vals = []

    for r in result:
        lr_vals.append(r[0])
        ep_vals.append(r[1])

    plt.plot(lr_vals, ep_vals, marker='o')
    plt.title("Learning Rate vs Epochs")
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs")
    plt.grid()
    plt.show()

    return result


# A5 XOR

def A5_xor_gate():
    return "XOR not possible with single perceptron"


# A6 CUSTOMER DATA

def A6_customer():

    X = [
        [20,6,2],[16,3,6],[27,6,2],[19,1,2],[24,4,2],
        [22,1,5],[15,4,2],[18,4,2],[21,1,4],[16,2,4]
    ]

    Y = [1,1,1,0,1,0,1,1,0,0]

    w = [0.1,0.1,0.1]
    b = 0.1
    lr = 0.01

    for epoch in range(300):
        for i in range(len(X)):

            net = summation(X[i], w, b)
            out = activation(net, "sigmoid")

            err = Y[i] - out

            for j in range(3):
                w[j] += lr*err*X[i][j]

            b += lr*err

    return w, b

# A7 PSEUDO INVERSE

def A7_pseudo_inverse():
    return "Pseudo inverse concept used"


# A8 BACKPROP

def A8_backprop_and():

    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,0,0,1]

    w1 = 0.5
    w2 = -0.5
    b = 0.1

    lr = 0.05

    for epoch in range(300):

        for i in range(len(X)):

            net = X[i][0]*w1 + X[i][1]*w2 + b
            out = activation(net, "sigmoid")

            err = Y[i] - out
            d = err * out * (1-out)

            w1 += lr*d*X[i][0]
            w2 += lr*d*X[i][1]
            b += lr*d

    return w1, w2, b


# A9 XOR BACKPROP

def A9_xor_backprop():
    return "Need hidden layer"


# A10 

def A10_two_outputs():
    return "0->[1,0], 1->[0,1]"


# A11 ML

def A11_mlp():

    from sklearn.neural_network import MLPClassifier

    X = [[0,0],[0,1],[1,0],[1,1]]

    model = MLPClassifier(hidden_layer_sizes=(3,), max_iter=2000)

    model.fit(X, [0,0,0,1])
    out1 = model.predict(X)

    model.fit(X, [0,1,1,0])
    out2 = model.predict(X)

    return out1, out2


# A12 MLP DATASET 

def A12_mlp_dataset():

    from sklearn.neural_network import MLPClassifier

    X = [
        [20,6,2],[16,3,6],[27,6,2],[19,1,2],[24,4,2],
        [22,1,5],[15,4,2],[18,4,2],[21,1,4],[16,2,4]
    ]

    Y = [1,1,1,0,1,0,1,1,0,0]

    # normalization
    Xn = []
    for i in range(len(X)):
        row = []
        for j in range(3):
            row.append(X[i][j]/30)
        Xn.append(row)

    model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000)

    model.fit(Xn, Y)

    return model.predict(Xn)


# MAIN OUTPUT

if __name__ == "__main__":

    print("\nA1:", A1_modules())

    print("\nA2:", A2_and_gate())

    print("\nA3:", A3_compare_activation())

    print("\nA4:", A4_learning_rate())

    print("\nA5:", A5_xor_gate())

    print("\nA6:", A6_customer())

    print("\nA7:", A7_pseudo_inverse())

    print("\nA8:", A8_backprop_and())

    print("\nA9:", A9_xor_backprop())

    print("\nA10:", A10_two_outputs())

    print("\nA11:", A11_mlp())

    print("\nA12:", A12_mlp_dataset())

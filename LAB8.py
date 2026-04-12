import math

# A1 
def A1_modules():

    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,0,0,1]

    w = [0.2, -0.75]
    b = 10

    outputs = []

    for i in range(4):

        # summation
        net = 0
        for j in range(2):
            net = net + X[i][j] * w[j]
        net = net + b

        # step activation
        if net >= 0:
            out = 1
        else:
            out = 0

        # error
        error = Y[i] - out

        outputs.append((net, out, error))

    return outputs



# A2 

def A2_and_gate():

    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,0,0,1]

    w = [0.2, -0.75]
    b = 10
    lr = 0.05

    epoch = 0

    while epoch < 1000:

        total_error = 0

        for i in range(4):

            net = 0
            for j in range(2):
                net = net + X[i][j] * w[j]
            net = net + b

            if net >= 0:
                out = 1
            else:
                out = 0

            error = Y[i] - out
            total_error = total_error + error * error

            for j in range(2):
                w[j] = w[j] + lr * error * X[i][j]

            b = b + lr * error

        if total_error <= 0.002:
            break

        epoch = epoch + 1

    return w, b, epoch


# A3 

def A3_compare_activation():

    activations = ["bipolar", "sigmoid", "relu"]
    results = []

    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,0,0,1]

    for act in activations:

        w = [0.2, -0.75]
        b = 10
        lr = 0.05
        epoch = 0

        while epoch < 1000:

            total_error = 0

            for i in range(4):

                net = 0
                for j in range(2):
                    net = net + X[i][j] * w[j]
                net = net + b

                if act == "bipolar":
                    if net >= 0:
                        out = 1
                    else:
                        out = -1

                elif act == "sigmoid":
                    out = 1 / (1 + math.exp(-net))

                elif act == "relu":
                    if net > 0:
                        out = net
                    else:
                        out = 0

                error = Y[i] - out
                total_error = total_error + error * error

                for j in range(2):
                    w[j] = w[j] + lr * error * X[i][j]

                b = b + lr * error

            if total_error <= 0.002:
                break

            epoch = epoch + 1

        results.append((act, epoch))

    return results


# A4 LEARNING RATE TEST 

def A4_learning_rate():

    rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    results = []

    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,0,0,1]

    for lr in rates:

        w = [0.2, -0.75]
        b = 10
        epoch = 0

        while epoch < 1000:

            total_error = 0

            for i in range(4):

                net = 0
                for j in range(2):
                    net = net + X[i][j] * w[j]
                net = net + b

                if net >= 0:
                    out = 1
                else:
                    out = 0

                error = Y[i] - out
                total_error = total_error + error * error

                for j in range(2):
                    w[j] = w[j] + lr * error * X[i][j]

                b = b + lr * error

            if total_error <= 0.002:
                break

            epoch = epoch + 1

        results.append((lr, epoch))

    return results


# A5 XOR DATASET 

def A5_xor_gate():

    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,1,1,0]

    w = [0.2, -0.75]
    b = 10
    lr = 0.05

    epoch = 0

    while epoch < 1000:

        total_error = 0

        for i in range(4):

            net = 0
            for j in range(2):
                net = net + X[i][j] * w[j]
            net = net + b

            if net >= 0:
                out = 1
            else:
                out = 0

            error = Y[i] - out
            total_error = total_error + error * error

            for j in range(2):
                w[j] = w[j] + lr * error * X[i][j]

            b = b + lr * error

        epoch = epoch + 1

    return total_error, epoch


# A6 CUSTOMER DATA 

def A6_customer():

    X = [
        [20,6,2],
        [16,3,6],
        [27,6,2],
        [19,1,2],
        [24,4,2],
        [22,1,5],
        [15,4,2],
        [18,4,2],
        [21,1,4],
        [16,2,4]
    ]

    Y = [1,1,1,0,1,0,1,1,0,0]

    w = [0.1,0.1,0.1]
    b = 0.1
    lr = 0.01

    epoch = 0

    while epoch < 500:

        for i in range(len(X)):

            net = 0
            for j in range(3):
                net = net + X[i][j] * w[j]
            net = net + b

            out = 1 / (1 + math.exp(-net))

            error = Y[i] - out

            for j in range(3):
                w[j] = w[j] + lr * error * X[i][j]

            b = b + lr * error

        epoch = epoch + 1

    return w, b


if __name__ == "__main__":

    print("A1 Output:")
    print(A1_modules())

    print("A2 AND Gate:")
    print(A2_and_gate())

    print("A3 Activation Comparison:")
    print(A3_compare_activation())

    print("A4 Learning Rate:")
    print(A4_learning_rate())

    print("A5 XOR Gate:")
    print(A5_xor_gate())

    print("A6 Customer Data:")
    print(A6_customer())

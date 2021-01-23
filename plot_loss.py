import matplotlib.pyplot as plt

input_file = './out28.log'


def process_lines(f: open) -> ([float]):
    x = []
    d = []
    g = []
    for line in f:
        line = line.rstrip()
        if line.startswith('Epoch'):
            _, epoch, _, time, _, d_loss, _, g_loss = line.split()
            epoch = epoch.split('/')[0]
            x.append(int(epoch))
            d.append(float(d_loss))
            g.append(float(g_loss))
    return x, d, g


if __name__ == '__main__':
    with open(input_file) as f:
        x, d, g = process_lines(f)
    plt.plot(x, d)
    plt.plot(x, g)
    plt.show()

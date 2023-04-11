if __name__ == "__main__":
    def decay(x):
        return 0.80 ** (1 / 1000000) * x

    epsilon = 0.99

    for i in range(1000000):
        epsilon = decay(epsilon)
        print(epsilon)
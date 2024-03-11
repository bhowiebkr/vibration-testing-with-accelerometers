import numpy as np
import pickle
import glob
import json
import numbers
import matplotlib.pyplot as plt
import statistics
from scipy.signal import savgol_filter


def combine_data(filename):
    new_data = {}
    for file in glob.glob("data_v2\\*"):
        with open(file) as f:
            d = json.load(f)

            for items in d:
                samples = d[items]
                samples_new = []

                for sample in samples:

                    num = True
                    for s in sample:
                        if not isinstance(s, numbers.Number):
                            print(f"{s} is not a number")
                            num = False
                    if num:
                        samples_new.append(sample)

                new_data[int(items)] = samples_new

    with open(f"{filename}.pickle", "wb") as f:
        pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_image(data, index=0, push_offset=0):
    steps = len(data)
    img = np.zeros(shape=(1000, steps), dtype=float)

    # Get the average
    first_step = str(sorted(list(map(int, data.keys())))[0])
    vals = []

    for d in data[first_step]:
        vals.append(d[index])
    zero_point = int(np.average(vals) * 100)

    for rpm in data:
        samples = data[rpm]

        for i in range(len(samples)):
            sample = samples[i]
            int_sample = int(sample[index] * 100)
            rpm_step = int(int(rpm) / 10.0) - 200

            try:
                img[int_sample + push_offset, rpm_step] += 1
            except IndexError:
                pass

    for i in range(steps):
        v_slice = img[:, i]
        v_slice = np.interp(v_slice, (v_slice.min(), v_slice.max()), (-1, +1))
        img[:, i] = v_slice

    return img, zero_point


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def graph_final():
    with open("SpindlePlate_10_rpm_500_samples.pickle", "rb") as handle:
        d = pickle.load(handle)

    data = {}
    for rpm in d:
        samples = d[rpm]

        x_samples = []
        y_samples = []

        for sample in samples:
            if not len(sample) == 3:
                continue

            x_samples.append(sample[0])
            y_samples.append(sample[1])

        mean = 40
        # statistics.median_grouped
        x_outliers_removed = reject_outliers(np.array(x_samples), mean)
        y_outliers_removed = reject_outliers(np.array(y_samples), mean)

        x = statistics.mode(x_outliers_removed)
        y = statistics.mode(y_outliers_removed)

        data[int(rpm)] = [x, y]

    x = []
    y = []
    z = []
    rpm = []

    for key in sorted(data):
        rpm.append(key)

        x.append(data[key][0])
        y.append(data[key][1])

    x_mode = x
    y_mode = y

    x = np.abs(x - (np.average(x)))
    y = np.abs(y - (np.average(y)))

    # smooth
    window_size = 5
    order = 1
    x = savgol_filter(x, window_size, order)
    y = savgol_filter(y, window_size, order)

    # normalize
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    x_mode = (x_mode - np.min(x_mode)) / (np.max(x_mode) - np.min(x_mode))
    y_mode = (y_mode - np.min(y_mode)) / (np.max(y_mode) - np.min(y_mode))

    a = (x + y) / 2
    a = savgol_filter(a, window_size, order)

    return rpm, x, y, a


def prep_data(filename):
    data = {}
    with open(f"{filename}.pickle", "rb") as handle:
        data = pickle.load(handle)

    x_img, x_zero_point = build_image(data, index=0)
    y_img, y_zero_point = build_image(data, index=1)
    print(f"x zero point: {x_zero_point}")
    print(f"y zero point: {y_zero_point}")

    # Slice the images
    w = 51
    offset = 5
    x_img = x_img[x_zero_point - w + offset : x_zero_point + w, :]
    w = 18
    offset - 5
    y_img = y_img[y_zero_point - w - 2 : y_zero_point + w, :]

    return x_img, y_img


def graph_data(data):
    x_img = data[0]
    y_img = data[1]
    rpm, x, y, a = graph_final()

    # X-Axis
    ax1 = plt.subplot(3, 1, 1)
    plt.title("2.2kw spindle with MPU-6050 accelerometer: (vibration magnitude/RPM)")
    plt.xticks(rpm[::20], rotation="vertical")
    plt.imshow(
        x_img,
        origin="lower",
        aspect="auto",
        cmap="jet",
        interpolation="antialiased",
        extent=(2000, 24000, 0, 1000),
    )

    ax1.axes.xaxis.set_ticks([])
    ax1.axes.xaxis.set_ticklabels([])
    ax1.grid(True, linestyle=":", axis="x")
    plt.ylabel("X-Axis")

    # Y-Axis
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.xticks(rpm[::20], rotation="vertical")
    plt.imshow(
        y_img,
        origin="lower",
        aspect="auto",
        cmap="jet",
        interpolation="antialiased",
        extent=(2000, 24000, 0, 1000),
    )

    ax2.axes.xaxis.set_ticks([])
    ax2.axes.xaxis.set_ticklabels([])
    ax2.grid(True, linestyle=":", axis="x")
    plt.ylabel("Y-Axis")

    # Combined
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    alpha = 0.5
    window_size = 5
    order = 3
    ax3.clear()
    plt.xticks(rpm[::20], rotation="vertical")
    plt.xlabel
    plt.plot(rpm, x, alpha=alpha, color="blue")
    plt.plot(rpm, y, alpha=alpha, color="green")
    plt.plot(rpm, a, color="red")
    plt.fill_between(rpm, a, color="red", alpha=alpha)
    plt.grid(True, linestyle="--")

    plt.ylabel("Resonance normalized")
    plt.xlabel("Spindle RPM")
    plt.legend(
        [
            "Accelerometer X-axis",
            "Accelerometer Y-axis",
            f"XY averaged & smoothed with: savgol_filter(window={window_size}, order={order})",
        ]
    )
    plt.show()


def main(filename):

    data = prep_data(filename)
    graph_data(data)


if __name__ == "__main__":
    filename = "SpindlePlate_10_rpm_500_samples"

    # combine_data(filename)
    main(filename)

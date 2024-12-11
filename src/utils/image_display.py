import matplotlib.pyplot as plt
import math


def show_image_from_dataset(dataset, fig_size: int = 10, images_to_show: int = 9):
    dataset_iterator = iter(dataset)
    plt.figure(figsize=(fig_size, fig_size))
    for i in range(images_to_show):
        plot_size = math.ceil(math.sqrt(images_to_show))
        plt.subplot(plot_size, plot_size, i + 1)
        batch = dataset_iterator.next()
        img = batch[0][i]
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    plt.show()
import matplotlib.pyplot as plt

def display_images(x1, x2, lbl, predicted_lbl):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(np.transpose(x1.cpu().numpy(), (1, 2, 0)))
    plt.title("Image 1")

    plt.subplot(1, 4, 2)
    plt.imshow(np.transpose(x2.cpu().numpy(), (1, 2, 0)))
    plt.title("Image 2")

    plt.subplot(1, 4, 3)
    plt.imshow(lbl.cpu().numpy(), cmap="gray")
    plt.title("True Label")

    plt.subplot(1, 4, 4)
    if isinstance(predicted_lbl, np.ndarray):
        plt.imshow(predicted_lbl, cmap="gray")  
    else:
        plt.imshow(predicted_lbl.cpu().numpy(), cmap="gray") 
    plt.title("Predicted Label")

    plt.show()

import torchvision
import torchvision.transforms as transforms


class MNIST:

    def __init__(self):
        super().__init__()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)

        self.testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)


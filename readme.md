# Neural Processes

> Implementation of Neural Processes [paper](https://arxiv.org/pdf/1807.01622.pdf) for _Projects in Machine Learning and Artificial Intelligence_ course at TU Berlin

## Requirements

- OS:
  - Convenience script [script/bootstrap](./script/bootstrap) supports only macOS or Linux.
  - On Windows, proceed to manually install the dependencies listed in [requirements/](./requirements/).
- `python3` (Tested on Python `3.7.3`)

## Installation

1. Execute [script/bootstrap](./script/bootstrap) to install project dependencies

   ```sh
   ./script/bootstrap
   ```

## Contributing

Code is formatted using [`autopep8`](https://pypi.org/project/autopep8/) to adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/).

- Format python files:

  ```sh
  ./script/format
  ```

## Related Projects

- [deepmind/neural-processes](https://github.com/deepmind/neural-processes) - Source code of Neural Processes [paper](https://arxiv.org/pdf/1807.01622.pdf)

## Credits

- This readme is based on [rodrigobdz/minimal-readme](https://github.com/rodrigobdz/minimal-readme).
- The [script](./script) structure is based on [rodrigobdz/styleguide-sh](https://github.com/rodrigobdz/styleguide-sh).


## Latest mnist loading workaround (29.03.2021)
from torchvision import datasets
new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [
   ('/'.join([new_mirror, url.split('/')[-1]]), md5)
   for url, md5 in datasets.MNIST.resources
]
train_data = datasets.MNIST(
   "./", train=True, download=True, transform = torchvision.transforms.ToTensor()
)
test_data = datasets.MNIST("./", train=True, download=True, transform = torchvision.transforms.ToTensor())
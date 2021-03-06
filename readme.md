# Neural Processes

> Implementation of Neural Processes [paper](https://arxiv.org/pdf/1807.01622.pdf) for _Projects in Machine Learning and Artificial Intelligence_ course at TU Berlin

**Disclaimer:** Our codebase is **strongly** inspired on [deepmind/neural-processes](https://github.com/deepmind/neural-processes) and we explicitely make use of some of their functions.

Authors: [@l8git](https://github.com/l8git) and [@rodrigobdz](https://github.com/rodrigobdz)

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

## Credits

- **[deepmind/neural-processes](https://github.com/deepmind/neural-processes)** - Source code of Neural Processes [paper](https://arxiv.org/pdf/1807.01622.pdf)
- This readme is based on [rodrigobdz/minimal-readme](https://github.com/rodrigobdz/minimal-readme).
- The [script](./script) structure is based on [rodrigobdz/styleguide-sh](https://github.com/rodrigobdz/styleguide-sh).

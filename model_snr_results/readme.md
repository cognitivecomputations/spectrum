# Model Results

This folder contains a collection of pre-scanned spectrum results for various models. If you find models that are not already included here, we encourage you to submit a Pull Request (PR) to add those results. We will continue to update this repository as we use spectrum in our models.

## Usage

```bash
python spectrum_analyzer.py --model-name <model-name> --top-percent <the top % of SNR modules you want to target>
```
## Contributing

We welcome contributions! If you would like to contribute:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new spectrum results for model XYZ'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Please make sure your contributions include:
- The JSON file containing the SNR results.
- A link to the Hugging Face (HF) repository the results came from.

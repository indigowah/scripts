# marching_converter

`marching_converter.py` is a Python script that batch converts `.docx`, `.pptx`, and `.pdf` files in a directory (and its subdirectories) to Obsidian-flavored Markdown files. The script preserves the original folder structure in the output directory.

## Features

- Converts Microsoft Word (`.docx`) files to Markdown using Pandoc.
- Converts PowerPoint (`.pptx`) slides to Markdown, with each slide as a section.
- Converts PDF files to Markdown using OpenAI's GPT model for formatting.
- Processes entire directory trees, replicating folder structure.
- Parallel conversion for faster processing.
- Progress reporting.

## Requirements

- Python 3.7+
- [pandoc](https://pandoc.org/) (for `.docx` conversion)
- [openai](https://pypi.org/project/openai/)
- [python-docx](https://pypi.org/project/python-docx/)
- [python-pptx](https://pypi.org/project/python-pptx/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)

Install dependencies with:

```sh
pip install -r requirements.txt
```

You must also have `pandoc` installed and available in your system PATH.

## OpenAI API Key

Set your OpenAI API key as an environment variable:

```sh
export OPENAI_API_KEY=your_openai_api_key
```

## Usage

```sh
python marching_converter.py <input_dir> <output_dir>
```

- `<input_dir>`: Path to the directory containing files to convert.
- `<output_dir>`: Path to the directory where Markdown files will be saved.

Example:

```sh
python marching_converter.py ./input_docs ./output_md
```

## Notes

- Only `.docx`, `.pptx`, and `.pdf` files are converted.
- PDF conversion uses OpenAI's GPT model, which may incur API costs and requires internet access.
- The script preserves the directory structure of the input folder in the output folder.

## License

MIT License
```

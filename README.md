This repo emulates the pipeline of API changes data collection for Spring libary from [the work for Rust](https://github.com/SYSUSELab/RustEvo). 

Evaluation and collection processes require `OPENAI_API_KEY` variable to be set in environment.

### Run evaluation
```
python -m evaluation.evaluate --help
usage: evaluate.py [-h] --input_file INPUT_FILE --output_file OUTPUT_FILE
                   [--models MODELS [MODELS ...]] [--max_workers MAX_WORKERS]
                   [--api_key API_KEY] [--base_url BASE_URL]
                   [--java_files_dir directory fo logs] [--type_info which info about API to include to the prompt (none, api)]
```

### Run data collection

```
python -m prepare_data.generate_query
python -m prepare_data.generate_code
python -m prepare_data.generate_tests
```

By default runs model `o1-mini` - set up environment variable `OPENAI_API_MODEL` to run other model


### Data
All API changes (between subsequent versions of Spring) are collected in `data/diffs` (`data/snapshots` describe the overall functionality by the version). `extractor` module is responsible for obtaining these changes.


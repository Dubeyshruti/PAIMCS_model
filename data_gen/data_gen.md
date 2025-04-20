#### Synthetic Text Data Generation for Upstream (pretraining) task
```bash
pip install -r data_gen_requirements.txt
python save_txt.py >> save_txt.out 2>&1
python data_gen.py en 2>&1 # partially done
python data_gen.py hi 2>&1  # to do
```



### Command I use to create interactive shell
```bash
interactive -c 10 -N 1 -G a100:2 -t 0-12:00
```

### My command to setup python environment
```bash
mamba create -n sol-testing python=3.10
source activate sol-testing
pip install -r 
pip install torch==2.0.1
pip install transformers
pip install accelerate
pip install sentencepiece
```

### Code to perform distributed training with Transformers Trainer class with Fully Sharded Data Parallel (FSDP) training
```bash

```  
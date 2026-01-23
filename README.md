<div align="center">
<img src="image/logo.png" width="200px">

**Meituan LongCat Interaction Team**

***Interaction Safety Group***
</div>

# Attention-MoA

Official code of `Attention-MoA: Enhancing Mixture-of-Agents via Inter-Agent Semantic Attention and Deep Residual Synthesis`


![Framework](image/framework.png)


## Experimental Results

We evaluate Attention-MoA on three benchmarks: AlpacaEval 2.0, MT-Bench, and FLASK.

### AlpacaEval 2.0 & MT-Bench
![AlpacaEval 2.0](image/comparesion.png)

### MT-Bench Details
![MT-Bench](image/mtbench.png)

### FLASK Details
![FLASK](image/FLASK.png)


## Evaluation

### Preparation
intall requirements
```
cd alpaca_eval
pip install -e .
cd FastChat
pip install -e ".[model_worker,llm_judge]"
cd ..
```

### Evaluation on AlpacaEval2.0
```
bash eval_alpaca.sh
```

### Evaluation on MT-Bench
```
bash eval_mtbench.sh
```

### Evaluation on Flask
```
bash eval_flask.sh
```
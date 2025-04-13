# Neural Probabilistic Language Model (Portuguese Implementation)

Este repositório contém minha implementação do clássico paper [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), de Yoshua Bengio et al.

Ao contrário da maioria das implementações disponíveis, que usam o corpus em inglês, aqui o modelo foi treinado usando **dados em português da Wikipedia** — o que traz desafios e vocabulário bastante diferentes!

## Resultados

### 🧠 O modelo aprendeu!
- **Loss de validação** caiu de **4.99 → 4.43** em 10 épocas.
- **Similaridade semântica** entre “rei” e “rainha” aumentou de **-0.03 → 0.21**.
- Palavras mais próximas de “rei” ao final do treinamento incluíam:  
  `"imperador"`, `"papa"`, `"substituiu"` — o que mostra que o modelo aprendeu relações contextuais coerentes.

### ⚙️ Treinamento
- Executado em CPU (sim, foi na força da vontade rsrs)
- 10 épocas sobre o dataset tokenizado da Wikipedia em português.
- Vocab size, dimensão de embeddings e demais parâmetros configuráveis no script.

## Requisitos

- Python 3.8+
- PyTorch
- tqdm
- numpy
- nltk

Instale os requisitos com:

```bash
pip install -r requirements.txt

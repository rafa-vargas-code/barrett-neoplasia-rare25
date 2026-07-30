# Classificação de Neoplasia em Esôfago de Barrett (RARE25)

Estudo comparativo de estratégias de deep learning para detecção de neoplasia em imagens endoscópicas do esôfago de Barrett, sob forte desbalanceamento de classes. Protocolo experimental com validação em duas camadas, teste estatístico e explicabilidade.

## Problema

Adenocarcinoma esofágico tem alta mortalidade; a detecção precoce de neoplasia no esôfago de Barrett é crucial. O dataset é fortemente desbalanceado — a classe positiva (neoplasia) é rara —, o que torna a acurácia enganosa e exige métricas e protocolo adequados.

## Dataset

[RARE25-train](https://huggingface.co/datasets/TimJaspersTue/RARE25-train) (HuggingFace) — 3095 imagens endoscópicas, classificação binária:

| Classe | Rótulo | Imagens | % |
|---|---|---|---|
| Não-displásico (ndbe) | 0 | 2937 | 94,9% |
| Neoplasia (neo) | 1 (positiva) | 158 | 5,1% |

Desbalanceamento 18,6:1.

## Pipeline

Notebook único (`app.ipynb`), 22 blocos documentados. Documentação bloco a bloco em [`DOCUMENTACAO_PIPELINE.md`](DOCUMENTACAO_PIPELINE.md).

**Eixos experimentais (varredura fatorial, 27 configurações):**
- Arquiteturas: ResNet50, EfficientNet-B0, MobileNetV2 (pré-treinadas, via `timm`)
- Funções de perda: CrossEntropy, CE ponderada `N/(2·Nc)`, Focal Loss (γ=2)
- Regularização: nenhuma, Mixup, CutMix

**Validação em duas camadas** (separa as fontes de variância):
- **Camada A — Fase 1:** 5-Fold estratificado sobre as 27 configs (135 treinos). Mede variância de partição.
- **Camada B — Fase 2:** split fixo 80/10/10, top-5 configs × 5 sementes (25 treinos). Isola variância de semente.

**Avaliação:** AUC-PR (métrica principal, adequada a desbalanceamento), AUC-ROC, F1, sensibilidade, especificidade. Teste de **Wilcoxon** pareado entre as melhores. Comparação com o baseline dos autores do RARE25. **Grad-CAM** com enquadramento técnico (sem overclaim clínico).

## Resultados

**Melhor configuração:** ResNet50 + CE ponderada + Mixup — AUC-PR 0,737±0,031, AUC-ROC 0,944±0,011 (baseline RARE25: 0,87).

- Arquitetura é o fator mais decisivo (ResNet50 domina); regularização vem depois (CutMix > Mixup > nenhuma); a função de perda importa pouco.
- Wilcoxon: as três melhores configs são estatisticamente **indistinguíveis** (p > 0,05).
- Variância de partição (Fase 1, std 0,09–0,16) >> variância de semente (Fase 2, std 0,03–0,07) — evidência de que um split único engana.

Resultados completos em [`resultados_recuperados/`](resultados_recuperados/).

## Stack

Python, PyTorch, timm, torchvision, HuggingFace datasets, scikit-learn, scipy, pytorch-grad-cam. Execução em GPU (Colab A100).

## Como executar

Abrir `app.ipynb` no Google Colab (runtime GPU). O dataset é baixado do HuggingFace no Bloco 2; o token é solicitado via `getpass` no Bloco 1. Rodar os blocos em ordem.

## Limitações

- Classe positiva pequena (158 imagens) limita a generalização; validação externa é trabalho futuro.
- Contribuição centrada em comparação rigorosa de fine-tuning + protocolo de validação, não em arquitetura nova.

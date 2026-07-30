'

# Documentação do Pipeline — Classificação de Neoplasia em Esôfago de Barrett (RARE25)

Registro do que cada bloco do notebook `app.ipynb` faz. Alvo: CBIAS 2026, possível revista depois.
Pipeline planejado: 5-Fold CV (Fase 1, 27 configs) + multi-seed 5 seeds (Fase 2, top 5) + Wilcoxon + Grad-CAM.

## Nota de performance (importante)

Bloco 8 pre-carrega todas as imagens redimensionadas (224x224) em RAM (`CACHE_IMG`) — evita re-decodificar do dataset HF a cada epoca, que era o gargalo (GPU ociosa, ~30 min/treino). Bloco 11 usa `batch_size=64`, `num_workers=8`, `pin_memory=True`. Transforms sem `Resize` (cache ja redimensiona). Alvo: ~1-3 min/treino na A100.

---

Convenção: cada bloco = célula markdown de título + célula de código, sem comentários `#` no código (explicação fica na markdown e aqui).

---

## Bloco 1 — Autenticação no HuggingFace

Solicita o token do HuggingFace via `getpass` (uma vez por sessão) e guarda em `os.environ["HF_TOKEN"]`. Necessário para baixar o RARE25.

**Status:** executado.

---

## Bloco 2 — Instalação e carregamento do RARE25

Instala `datasets`, desativa o backend Xet do HuggingFace (`HF_HUB_DISABLE_XET=1`) forçando download HTTP padrão, aumenta o timeout, e carrega o dataset com retry + backoff exponencial (até 8 tentativas) para tolerar falhas de rede. Inclui referência bibliográfica formal do dataset (corrige crítica do Revisor 2 sobre reprodutibilidade).

**Status:** executado — dataset carregado.

---

## Bloco 3 — Verificação rápida do dataset

Confirma o carregamento: número de imagens no split `train`, chaves de cada exemplo, valor do rótulo e tamanho da primeira imagem.

**Status:** executado.

---

## Bloco 4 — Exploração do dataset

Distribuição de classes com `Counter` (confirma desbalanceamento ~5% neoplasia), inspeção do mapeamento do rótulo (`features['label']`, qual índice é a classe positiva) e contagem dos tamanhos distintos de imagem (justifica o resize para 224x224).

**Status:** executado. Resultados:

- Classe 0 = `ndbe` (negativa): 2937 imagens (94,9%)
- Classe 1 = `neo` (positiva): 158 imagens (5,1%)
- Total 3095, desbalanceamento 18,6:1
- 83 tamanhos distintos, altura fixa 512, largura 587–655
- Peso de classe `N/(2·Nc)`: w0≈0,53, w1≈9,79

---

## Bloco 5 — Exemplos visuais das classes

Grade 2×4 com imagens `ndbe` vs `neo` lado a lado, salva em `exemplos_classes.png`. Figura para a Introdução (pedida pelos três revisores do SBCAS).

**Status:** executado.

---

## Bloco 6 — Particionamento em duas camadas

**Camada A (Fase 1):** `StratifiedKFold` (5 folds, `random_state=42`) sobre todo o dataset. Cada config treinada 5× (test em fold diferente), usa 100% dos dados. Mede variância de partição. ~31 neoplasias por fold de teste.

**Camada B (Fase 2):** split fixo estratificado 80/10/10 (`random_state=42`), usado só nas top-5 configs × 5 seeds. Partição imutável isola a variância de semente da variância de partição.

Decisão confirmada pelo usuário: seguir com as duas camadas.

**Status:** executado. Camada A: 5 folds com 31–32 neo por teste (~5%). Camada B: treino 126 / val 16 / teste 16 neo (~5%).

---

## Bloco 7 — Caracterização das arquiteturas

Tabela comparativa de ResNet50, EfficientNet-B0, MobileNetV2: nº de parâmetros (M), dimensão do vetor de features, tempo de inferência (ms/img, medido na GPU com warmup) e FLOPs (via `fvcore`). Salva `arquiteturas.csv`. Responde à crítica de "seleção arbitrária" das arquiteturas.

**Status:** executado (A100 80GB). Tabela final:

| Arquitetura     | Params (M) | Dim. features | FLOPs (G) | Inferência (ms/img) |
| --------------- | ---------- | ------------- | --------- | -------------------- |
| ResNet50        | 23,51      | 2048          | 4,11      | 6,52                 |
| EfficientNet-B0 | 4,01       | 1280          | 0,40      | 8,81                 |
| MobileNetV2     | 2,23       | 1280          | 0,31      | 6,04                 |

Achado: EfficientNet-B0 é a mais lenta apesar de 10× menos FLOPs — convoluções depthwise-separable subutilizam a GPU em batch=1. FLOPs não prevê latência real.

---

## Bloco 8 — Transforms e classe Dataset

Pré-carrega todas as imagens (RGB, 224×224) em RAM (`CACHE_IMG`) uma única vez. `train_transform`: augmentation (flip, rotação ≤15°, ColorJitter) + normalização ImageNet (sem `Resize`, cache já redimensiona). `eval_transform`: só normalização. Classe `BarrettDataset(indices, transform)` lê do cache e retorna `(tensor, label)`.

**Status:** executado com cache. `CACHE_IMG` reduziu treino de ~30 min para ~3 min.

---

## Bloco 9 — Funções de perda

Três eixos de loss, pesos calculados dinamicamente (sem hardcode do `126` original): `ce` (entropia cruzada padrão), `weighted` (CE ponderada por `N/(2·Nc)`), `focal` (Focal Loss, γ=2, com pesos como α — Lin et al. 2017). Função `criar_criterion(nome, y_train, device)` devolve o critério com os pesos da partição atual.

**Status:** executado. w0=0,527, w1=9,825.

---

## Bloco 10 — Mixup e CutMix

Terceiro eixo (regularização). `mixup_data` (mistura linear, λ~Beta), `cutmix_data` (recorte retangular), `mixup_criterion` (perda ponderada pelos dois rótulos). Refs: Zhang 2018, Yun 2019.

**Status:** executado.

---

## Bloco 11 — Reprodutibilidade e DataLoaders

`set_seed` (Python/NumPy/Torch/CUDA/cuDNN determinístico), `seed_worker`, `criar_dataloaders` (com `pin_memory`, `num_workers=8`), `CONFIG` (50 épocas, paciência 15, lr 1e-4, **batch 64**). Doc no bloco lista o que a semente controla com pesos pré-treinados (ordem batches, λ Mixup, augmentations, dropout, caminho early stopping) — resposta direta ao Revisor 2.

**Status:** executado.

---

## Bloco 12 — Métricas e avaliação

`avaliar` (probabilidades da classe positiva) e `calcular_metricas` (AUC-PR principal, AUC-ROC, F1, sensibilidade, especificidade, precisão). Limiar 0,5.

**Status:** executado.

---

## Bloco 13 — Função principal de treinamento

`treinar_modelo(...)`: fixa seed, cria loaders, modelo pré-treinado, criterion com pesos da partição, Adam. Treina até 50 épocas com early stopping na AUC-PR de val (paciência 15), aplica Mixup/CutMix por batch. Restaura melhores pesos, avalia no teste.

**Status:** executado.

---

## FASE 1 — Bloco 14: Varredura 27 configs × 5-Fold

3 arqs × 3 losses × 3 regs × 5 folds = 135 treinos. 15% do treino de cada fold vira val (early stopping). Seed fixa 42. Salva incremental em `resultados_fase1.csv` com retomada. Job longo (horas na A100).

**Status:** executado (135 treinos, A100, ~3 min/treino). Achado: ResNet50 domina as 9 primeiras posições; EfficientNet-B0 só aparece na 10ª. Modelo maior vence a tarefa desbalanceada. `resultados_fase1.csv` recuperado dos outputs do notebook após reciclagem do VM.

---

## FASE 1 — Bloco 15: Resumo e heatmap

Média±desvio da AUC-PR por config entre folds, `resumo_fase1.csv`, heatmap arquitetura×loss (`heatmap_fase1.png`). Define as top-5 da Fase 2.

**Status:** executado. Top-5 (AUC-PR média±desvio):

| # | Arq | Loss | Reg | AUC-PR |
|---|---|---|---|---|
| 1 | ResNet50 | weighted | cutmix | 0,731±0,092 |
| 2 | ResNet50 | weighted | mixup | 0,715±0,093 |
| 3 | ResNet50 | focal | cutmix | 0,709±0,097 |
| 4 | ResNet50 | ce | cutmix | 0,695±0,071 |
| 5 | ResNet50 | focal | none | 0,674±0,121 |

Variância entre folds alta (std 0,09–0,16) — só ~31 neo por fold de teste. Justifica os 5 folds e mostra que número único de split engana (munição contra o motivo da rejeição SBCAS). AUC-ROC/F1 dos 17 configs fora do top-10 perdidos na reciclagem do VM (só AUC-PR recuperável dos 27).

---

## FASE 1 — Bloco 16: Análise por fator

Efeito médio marginal de arquitetura, loss e regularização na AUC-PR. Base da discussão.

**Status:** executado. Fatores:
- **Arquitetura** é o mais decisivo (ResNet50 >> EfficientNet/MobileNet).
- **Regularização tem efeito de interação com a arquitetura** (não é efeito marginal simples): na média das 3 arquiteturas, `none` vence (0,577 > mixup 0,507 > cutmix 0,504) porque Mixup/CutMix degradam as redes menores; mas **dentro da ResNet50** a ordem inverte (cutmix 0,712 > mixup 0,681 > none 0,642). MobileNetV2 sofre forte com regularização forte (focal+cutmix cai a AUC-PR 0,16–0,28).
- **Loss** importa pouco (weighted/focal/ce próximas no topo).

Cuidado ao redigir o CBIAS: NÃO afirmar "cutmix > mixup > none" como efeito geral — só vale para a ResNet50. O achado correto é a **interação** arquitetura×regularização. Discussão honesta: loss ponderada não é bala de prata (responde crítica Rev3).

---

## FASE 2 — Bloco 17: Multi-seed das top-5

Top-5 × 5 seeds (42,7,123,0,2024) no split fixo 80/10/10 = 25 treinos. Isola variância de semente. `resultados_fase2.csv` com retomada.

**Status:** executado (25 treinos). `resultados_fase2.csv` recuperado dos outputs do notebook.

---

## FASE 2 — Bloco 18: Média±desvio por config

Agrega multi-seed (AUC-PR, AUC-ROC, F1, sens, espec), `resumo_fase2.csv`. Tabela principal do artigo.

**Status:** executado. Tabela principal (AUC-PR média±desvio entre 5 seeds):

| Arq | Loss | Reg | AUC-PR | AUC-ROC | Sens | Espec | F1 |
|---|---|---|---|---|---|---|---|
| ResNet50 | weighted | mixup | 0,737±0,031 | 0,944 | 0,775 | 0,925 | 0,500 |
| ResNet50 | ce | cutmix | 0,737±0,068 | 0,914 | 0,550 | 0,994 | 0,663 |
| ResNet50 | focal | cutmix | 0,705±0,047 | 0,913 | 0,825 | 0,807 | 0,321 |
| ResNet50 | weighted | cutmix | 0,693±0,039 | 0,909 | 0,675 | 0,965 | 0,597 |
| ResNet50 | focal | none | 0,692±0,055 | 0,895 | 0,738 | 0,896 | 0,456 |

`weighted mixup` e `ce cutmix` empatam no topo (0,737). Desvio de semente baixo (0,03–0,07) vs desvio de partição da Fase 1 (0,09–0,16) — separar as duas fontes de variância era o ponto das duas camadas. F1 baixo do focal reflete descalibração no limiar 0,5 (ranqueia bem em AUC-PR, mas empurra probabilidades pra baixo).

---

## FASE 2 — Bloco 19: Teste de Wilcoxon

Top-3 duas a duas, Wilcoxon pareado por seed na AUC-PR. p>0,05 = indistinguíveis. Teste formal exigido por revisor.

**Status:** executado. Top-3 (weighted mixup, ce cutmix, focal cutmix) todas **indistinguíveis**: p=1,000 / 0,3125 / 0,8125. Nenhuma diferença estatística significativa entre as melhores — resultado honesto, alinhado ao que o próprio artigo original relatou.

---

## FASE 2 — Bloco 20: Comparação baseline RARE25

AUC-ROC da melhor config vs baseline dos autores (0,87).

**Status:** executado. Melhor config (ResNet50 weighted mixup) AUC-ROC 0,944±0,011 vs baseline RARE25 0,87 = **+0,074**. Supera o baseline dos autores.

---

## FASE 3 — Bloco 21: Grad-CAM

Re-treina melhor config (seed 42), mapas Grad-CAM em neoplasias do teste (`gradcam_neoplasia.png`). Enquadramento técnico sem overclaim clínico (corrige crítica). Camada-alvo: `layer4[-1]` (ResNet) ou `blocks[-1]`.

**Status:** executado. Config usada: ResNet50 weighted mixup, seed 42. `gradcam_neoplasia.png` gerado (4 neoplasias do teste). Config cravada no bloco (não lê mais `resumo_fase2.csv`) para rodar standalone após reinício de kernel.

---

## FASE 4 — Bloco 22: Exportação

Monta Drive, copia CSVs e figuras para `MyDrive/RARE25_resultados`.

**Status:** executado. Exportado para `MyDrive/RARE25_resultados`. Os 5 arquivos perdidos na reciclagem do VM (resultados/resumo das duas fases + heatmap) foram reconstruídos a partir dos outputs salvos no `.ipynb` e enviados ao Drive.

---

## Nota de recuperação (reciclagem do VM Colab)

Após reinício do PC, o VM do Colab reciclou e apagou o `/content` antes do export ao Drive. Todos os números foram recuperados dos outputs salvos nas células do `app.ipynb`: Fase 1 (135 treinos), Fase 2 (25 treinos), Wilcoxon. CSVs reconstruídos em `resultados_recuperados/` e no Drive. Único ponto irrecuperável: AUC-ROC/F1 dos 17 configs da Fase 1 fora do top-10 (o print agregava só top-10; AUC-PR dos 27 recuperado do raw). Lição: salvar resultados direto no Drive, não no `/content` volátil.

---

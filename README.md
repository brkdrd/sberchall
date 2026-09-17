Российский квантовый центр выступает ведущим научно-технологическим хабом страны, где фундаментальные исследования в области квантовой физики трансформируются в конкретные индустриальные решения. РКЦ представляет собой уникальную экосистему из 17 лабораторий и высокотехнологичных стартапов, базирующуюся в инновационном центре «Сколково». Здесь в рамках дорожной карты «Квантовые вычисления» создаются мощнейшие отечественные квантовые процессоры на различных платформах, а также разрабатываются алгоритмы квантовых вычислений и квантового машинного обучения, способные в будущем радикально ускорить работу нейросетей и решение сложнейших задач оптимизации. Участие центра в олимпиаде открывает молодым исследователям доступ к передовому опыту разработки квантового программного обеспечения и позволяет прикоснуться к технологиям, которые прямо сейчас меняют облик современных вычислений.

Описание задачи основного этапа
Квантовый приближенный алгоритм оптимизации (QAOA) использует гибридную архитектуру. Однако классический цикл оптимизации, отвечающий за поиск параметров (углов), требует тысяч запусков квантовой схемы, что является вычислительным узким местом. Поэтому применение ML-моделей для предсказания оптимальных углов по входным данным задачи является актуальным направлением, что позволяет радикально сократить время работы алгоритма. Если не решать эту задачу, масштабирование квантовых вычислений на реальные индустриальные проблемы будет сильно ограничено сходимостью классических оптимизаторов.

Область: Машинное обучение в квантовых вычислениях (Quantum ML).

Тип ML-задачи: Множественная регрессия (Multi-output regression).

Дано на вход: Фиксированная матрица квадратичных взаимодействий (12-кубитная модель Изинга), вектор линейных коэффициентов и Python-симулятор.
 

Что нужно предсказать: 
<img width="1780" height="380" alt="image" src="https://github.com/user-attachments/assets/c65b57a3-e8cc-471b-a41f-2eb261e701a9" />

Для каждого вектора линейных коэффициентов из тестовой выборки – оптимальные углы gamma и beta для схемы QAOA глубины 5 (по 5 углов каждого типа). Решение оформляется в виде csv файла: с углами gamma и с углами beta

Решения будут оцениваться на следующей метрике:


 



Данные
QAOA.py – Python-симулятор симулятор схемы QAOA: по (h, gamma, beta) при фиксированной J считает квантовое состояние и метрику
J.npy – фиксированная матрица квадратичных взаимодействий модели Изинга, форма (12, 12).
h_train.npy – обучающая выборка с векторами линейных коэффициентов, форма (500, 12). Целевые углы (gamma, beta) участники генерируют самостоятельно, и сдают в тестирующую систему.
h_test.npy – тестовая выборка с векторами линейных коэффициентов, форма (500, 12). Целевых углов в файле нет. Выдается за 2 дня до завершения основного этапа, 15 сентября до 18:00 (по МСК). 
submission.csv - пример посылки со случайными углами
Требования к формату решения
Участникам рекомендуется представить следующие файлы решения:

Для подсчета метрики (автоматической проверки решения):
submission.csv — один файл из 500 строк (по строке на каждый вектор h_test) с колонками id, gamma_0, …, gamma_4, beta_0, …, beta_4: предсказанные углы γ и β для схемы QAOA глубины 5.
Для экспертной проверки решения:
solution.ipynb или zip-архив с кодом проекта – код с генерацией обучающей выборки, обучением модели и комментариями;
presentation.pdf – презентация решения;
README – описание данных и инструкции по запуску.
В течение основного этапа участникам доступны только J и h_train. Лидерборд во время основного этапа показывает метрику P(ground), посчитанную на обучающей выборке – это же значение участник может получить и локально, прогнав свою модель через QAOA.py. За 2 дня (15 сентября до 18:00 по МСК) до окончания основного этапа выкладывается h_test: участник применяет на нём готовую модель, получает предсказанные углы и сдаёт финальное решение (submission.csv + код + презентация). Итоговые баллы считаются по P(ground) на h_test.

Критерии оценки решения
1. Точность предсказания (по метрике) – до 70 баллов
<img width="1246" height="555" alt="image" src="https://github.com/user-attachments/assets/4551abc0-4eeb-4d36-b370-c00a9fd98caf" />
2. Качество и инновационность ML-решения – до 20 баллов

15-20 баллов: оригинальный подход к архитектуре нейросети.
8-14 баллов: стандартная архитектура (MLP), но проведена качественная работа с признаками и гиперпараметрами.
1-7 баллов: минимально рабочая архитектура без попыток оптимизации.
0 баллов: решение не прикреплено или не воспроизводится, или не соответствует задаче.
 
3. Качество оформления решения и воспроизводимость – до 10 баллов

10 баллов: структурированный код, наличие списка библиотек и подробное описание архитектуры и запуска в README.
5 баллов: код рабочий, но отсутствуют комментарии и описание логики решения.
0 баллов: решение невозможно запустить без ручной правки путей или установки специфических зависимостей, не указанных автором.
Результаты по данной задаче в рамках основного этапа будут учитываться при определении победителя в финальном этапе (победитель будет определяться путем суммирования баллов, полученных в основном этапе, финальном этапе и за онлайн-защиту).

Ограничения и требования к решению
Предсказание оптимальных углов для всего h_test.npy (инференс обученной модели) не должно превышать 10 минут.
Модель обязана использовать h_test как вход и выдавать углы в зависимости от h; сдача одинаковых (константных) углов для всех инстансов запрещена и оценивается в 0 баллов за весь этап.
Весь код должен воспроизводиться в Google Colab «в одну кнопку»: все инструкции по установке библиотек включены в решение, ноутбук исполняется от начала до конца без ручных правок. Если решение не запускается – 0 баллов за весь этап.
Инференс модели на h_test в Colab должен занимать не более 10 минут.
Обучение модели также должно работать в Colab. Оно может не укладываться в лимиты Colab по времени – это допустимо, но в решении обязаны быть предельно понятные инструкции по запуску обучения и описание модели.
В случае нарушения перечисленных требований, решение получит оценку 0 баллов за весь этап.

Выбор лучшего решения участником
Для каждого этапа Конкурса Участник самостоятельно определяет и отмечает итоговое Решение из числа загруженных, которое в дальнейшем будет передано Экспертам для оценки. Оценке подлежит то Решение, которое Участник отметил на Платформе как Лучшее (итоговое).

В рамках индивидуального трека Участник самостоятельно выбирает лучшее Решение на Платформе. В случае выбора итоговым нескольких Решений Участником, Организатор оценивает в рамках Конкурса последнее выбранное итоговое Решение. В случае отсутствия выбора Участником итогового решения Организатор оценивает в рамках Конкурса последнее загруженное Решение Участника.

---

# Solution: a mixture of angle experts

**Run it:** open `solution.ipynb` in Google Colab and `Runtime -> Run all`. It clones this
repo, takes `J.npy`, `h_test.npy` and the organisers' unmodified `QAOA.py` from it, loads
the trained checkpoint, and writes `submission.csv`. Nothing needs editing by hand, and the
inference cell prints its own wall clock against the 600 s limit rather than asserting it
fits.

## Why a codebook and a gate

Three measurements, not an architectural preference:

- **It is not regression.** For a fixed `h` the good angle vectors form several disjoint
  blobs, one per basin. An L2 fit to a set of search-generated labels lands on their mean,
  which lies in no basin at all.
- **It is not search.** 500 winning angle vectors cross-evaluated against all 500 instances
  give mean P 0.315 with *no search at all*, against 0.312 for the 16k-start search that
  produced them (`src/library.py`). Quadrupling the starts moved nothing.
- **It is selection.** The good basins are shared between instances; what changes with `h`
  is *which* one is right.

So: a learned codebook `C` of M angle vectors, and a gate that scores them against `h`.
Both are trained together, straight through the organisers' differentiable simulator, on

    L(h) = -log  sum_m  softmax(gate(h))_m * P_ground(h, C_m)

Maximising the mixture rather than the arg-max expert is what makes it trainable: every
expert receives gradient in proportion to the responsibility the gate assigns it, so
experts specialise while the gate partitions. It is the EM split, done by gradient descent
through a quantum circuit. No angle labels are generated anywhere; training `h` is
synthesised U(-1, 1) and the official `h_train` is validation only.

## The features fold the symmetry group away exactly

`P_ground` is invariant under a 128-element group: swapping the two qubits of any of the
six pairs `(i, 11-i)` (since `v_i = v_{11-i}` makes `J` blind to it), and `h -> -h`. All 128
leave the optimal angles unchanged, so feeding raw `h` asks the gate to learn 128 copies of
one function. `moe.canonical_features` folds it away **bit-exactly** — 0.0 deviation over
all 128 elements on both `h_train` and `h_test` — and three things had to be right for that:

- the sign flip and the pair sort **do not commute** (negating `h` exchanges min and max
  inside every pair), so the sign is fixed first;
- the sign key is `sum_i h_i v_i`, not the mean-field magnetisation: the latter is discrete
  and **exactly zero on 43 of the 500** `h_train` instances, which would leave those
  un-canonicalised;
- `sim.quad` is float32, so a configuration and its pair-swapped twin differ by ~1e-7. On
  the instance whose gap is 6.3e-5 that is a 2e-3 *relative* error landing straight on
  `log(gap)`, so the feature path rebuilds the energy table in float64.

## Running it

```bash
docker compose run --rm moe-smoke-cpu    # ~1 min, proves the code and the mounts
docker compose run --rm moe-smoke        # same on the GPU
docker compose run --rm moe              # the real training run
docker compose run --rm moe-validate     # the submission code path, scored on h_train
docker compose run --rm moe-predict      # runs/submission.csv for h_test
```

Every hyperparameter is a flag: `python -m src.moe --help`.

## What the p=5 ceiling actually is

Worth stating plainly, because it is the result of about twenty GPU-hours. Six independent
methods agree per instance, on instance 88 (spectral gap 0.025): Adam multistart 0.107, a
wide `|gamma| <= 12` box 0.092, coordinate-wise **global** grid search 0.104, continuation
in the coupling strength 0.107, energy-first then P 0.096, and tolerance annealing 0.104.
Growing the depth with INTERP gives 0.10 / 0.21 / 0.26 / 0.34 at p = 5 / 8 / 10 / 14.

The annealing run says it most clearly. Optimising `P(E <= Emin + tau)` and shrinking
`tau`, the circuit reaches **0.977** at `tau = 4` and loses almost all of it by `tau = 0`.
Depth 5 concentrates amplitude in the low-energy window perfectly well; what it cannot do
is resolve the ground state from a competitor 0.025 below it. That is why mean P over 500
instances sits near 0.32 and why the narrow-gap quartile does not move under 11x budget.

# Previous solution: the node-chain policy

A chain of **nodes**. A node is a point in angle space, and it owns its **surroundings**:
`k` probes sampled in a ball around it, each evaluated and then run forward under Adam.
That is what the model observes — not one number saying how good the node is, but what the
landscape does around it and which way the local flow runs.

```
root  <- RootMLP(h), explored
repeat:
    child 1 <- head + policy(head's surroundings)
    child 2 <- head + policy(head's surroundings, child 1's)
    child 3 <- head + policy(head's surroundings, child 1's, child 2's)
    head, parent, grandparent <- child 3, head, parent
```

The first two children are probes the policy places deliberately and then reads before
committing; the third is the commitment and becomes the next head. The policy sees three
generations back, which is what lets it recognise a direction it has already tried.

### Why REINFORCE and not backpropagation

Every node is `parent centre + emitted offset`, so a chain is a plain sequence of actions
and a pathwise gradient through it *is* available. It is still the wrong estimator. The
reward comes from the Adam finishes inside the surroundings, and `start -> finish` is
piecewise constant in value: its derivative is zero inside a basin and undefined at the
boundary between two. The event the policy has to learn is **which basin a node falls
into**, and that is exactly the event a pathwise gradient cannot see. So surroundings are
detached observations, the only gradient path is the action log-probs, and the terminal
reward — the best value in the final head's surroundings — is credited to the whole chain.

`chains` chains run per instance and each is scored against the mean of its siblings.
Leave-one-out, so the baseline is independent of the chain it corrects and the estimator
stays unbiased; a single terminal reward over ~13 actions does not train without one.

### Three choices that carry weight

- **Positions in tokens are relative to the head**, and so is the emitted offset, so a
  configuration of probes means the same thing wherever the head sits. The absolute
  position enters once, through the adaLN conditioning, because the landscape is *not*
  translation invariant — `gamma = 0` is a real place and the policy has to know where it
  is.
- **Roles are an embedding, not a sequence position.** The context is a set of probes
  tagged by which node owns them; ordering them would invent structure that is not in the
  data. A learned readout token replaces "take the last position" — and because context is
  addressed by role, neither the probe count nor the chain length is baked into the model,
  so **inference runs a bigger chain than training on the same weights**.
- **The offset head is zero-initialised** (adaLN-Zero, as in `model.py`), so an untrained
  chain is a random walk of scale `sigma` rather than a random jump of unbounded size: the
  run degrades to structured multistart instead of to noise.

### The control is part of the result

Most of a chain's compute is the Adam refinement inside its surroundings, and that
produces good angles whether or not a policy chose where to put them. Every evaluation
therefore also runs `random_control` at a matched forward-pass budget. The gap between the
two lines is the result; the chain's own number alone says nothing.

Code: `src/policy.py` (RootMLP + NodePolicy), `src/chain.py` (nodes, the exploration
operator, the cycle, tokenisation, the control), `src/reinforce.py` (training, evaluation,
submission), `src/refine.py` (the shared score/refine primitives).

`src/reinforce.py` also *measures* where the root should start rather than assuming it:
`probe_root_scale` scores a TQA schedule plus a short polish across `gamma_top` in
0.16..32 and starts the policy at the winner.

# Previous solution: learned-optimizer transformer

A decoder-only transformer conditioned on the instance vector `h` via **adaLN-Zero**
(DiT-style) that acts as a *learned optimiser* over QAOA angles. There are no angle
labels anywhere — the model is trained by backpropagating **through the differentiable
QAOA simulator** (`src/qaoa_ref.py`).

## How it works

### Angle normalisation (`src/angles.py`)

The two halves of the angle vector are **not** on the same scale, and the pipeline used to
treat them as one homogeneous 10-vector. The phase separator applies `exp(i*gamma*E)`, and
for this `J` with `h ~ U(-1,1)` the spectrum spans about **39**, so the phase completes a
full revolution by `gamma ~ 2*pi/39 ~ 0.16`. The mixer is periodic in `beta` with period
`pi` regardless of the problem. The two natural scales are therefore ~20x apart.

Ignoring that broke three things at once:

- **search** — sampling `gamma` uniformly on `(-pi, pi)` puts only `(0.16/pi)^5 ~ 3e-7` of
  the box in the region that carries signal; a 65k-point screen expects 0.02 useful hits;
- **step size** — one Adam `lr` is 2% of `beta`'s useful range but 19% of `gamma`'s;
- **model inputs** — `d logP/d gamma` carries a factor of `E ~ +-20` relative to
  `d logP/d beta`, so the shared `GRAD_CLIP` saturated the gamma gradient features and fed
  the model a dead input for exactly the coordinates that matter most.

Everything upstream of the simulator now works in normalised units `u`, with
`angles = u * ANGLE_SCALE`, where the scale is *measured* from the spectrum at startup
rather than assumed. A unit step means the same thing in both halves, and
`d logP/du = scale * d logP/d angle` puts the gradient features on a common magnitude by
construction. The scale travels inside the checkpoint (`angle_scale`); checkpoints saved
before this change carry none and fall back to the identity, reproducing the old behaviour
exactly so the two can be compared.

### The trajectory

Each sequence is an optimisation trajectory. A token is a 21-dim vector describing one
visited point in **normalised** angle space:

```
[ 10 units (5 gamma + 5 beta) | log P(ground) | d log P / d units (10) ]
```

- **Step 0**: random angles, evaluated by the simulator.
- **Each next step**: the transformer reads the whole trajectory so far (causal
  attention, `h` injected into every block through adaLN-Zero modulation) and outputs a
  **delta** added to the current angles. The new point is evaluated by the simulator,
  packed into a token, appended — and the model runs again. 8 steps per rollout.
- **Loss**: `-log P(ground)` of every predicted point, summed over the rollout with
  weights increasing toward later steps. The simulator is written in torch, so the
  gradient flows loss → simulator → predicted angles → transformer. No RL needed:
  tokens are continuous and the "environment" is differentiable.
- **Truncated BPTT**: history tokens are detached; gradient reaches each step's
  prediction only through its own loss. Exploration is Gaussian noise on the predicted
  angles (reparameterised, decaying over training).
- **Training data is free**: `h_train` is i.i.d. U(-1,1), so fresh instances are
  synthesised every iteration; the official `h_train` is held out for evaluation.
- **Inference**: the landscape is multi-modal, so we run K parallel rollouts from
  random starts per instance and keep the best point of the best trajectory
  (best-of-K), optionally polished by a few Adam steps through the simulator.

Code layout: `src/experiment.py` (**entry point** — `python -m src`, and its `CONFIG` dict is
the run's configuration), `src/model.py` (transformer), `src/angles.py` (angle normalisation
and the symmetry canonicaliser), `src/rollout.py` (sim-in-the-loop rollout), `src/train.py`,
`src/validate.py`, `src/predict.py`, `src/basins.py` + `src/proposer.py` (learned restarts,
below), `src/qaoa_ref.py` (differentiable simulator — the organisers' file, unmodified).

## Trust-region Bayesian optimisation (`src/turbo.py`)

TuRBO (Eriksson et al., NeurIPS 2019) fits a *local* GP inside a trust region around the
best point found, picks candidates by Thompson sampling, and grows or shrinks that region
on streaks of success or failure — restarting it elsewhere when it collapses. A single
global GP over-smooths a landscape this rugged and ends up sampling its own mean.

The implementation runs `N x n_tr` trust regions batched on the GPU, and adds four things
to stock TuRBO that are specific to this problem: the initial design is seeded from the
annealing-schedule family rather than pure Sobol; the box covers exactly one `beta` period
so the search is not re-exploring 32 copies of every optimum; the objective is `log
P(ground)`, which spans four orders of magnitude; and the final point gets an Adam polish,
since refusing to use exact gradients we already have would be an affectation. Matched-budget
`tqa` and `uniform` controls run alongside, because a score without them is uninterpretable.

No training stage — `python -m src` with `mode="turbo"` is a single pass over `h`.

## Learned restarts (`src/basins.py`, `src/proposer.py`)

A second mode, and the one `CONFIG["mode"]` selects by default. The premise is that Adam
through the simulator is already the best angle-finder available — its only real failure is
converging into a bad basin — so the model's job is to say *where to start it*, not to
replace it.

- `src/basins.py` runs Adam forwards from many random starts per instance and keeps the
  starts that reached that instance's best-known optimum. Those are labelled points inside
  the good basin. (Running the flow *backwards* from a known optimum does not work: reverse
  time is expanding, so integration error blows up, and Adam is not a reversible map.)
- `src/proposer.py` trains `h -> K starts` against those labels with a Chamfer loss —
  each prediction is pulled to its *nearest* label, and each label pulls its nearest
  prediction. The minimum is what matters: for a fixed `h` the good starts form several
  disjoint blobs, and an ordinary L2 regression would fit their mean, which lies in no
  basin at all.
- Inference is unchanged otherwise: propose K starts, Adam from each, select after
  polishing. The run reports the same budget spent on *random* starts alongside, because
  that is the only comparison that says whether the model contributed anything.

Both stages lean on `angles.canonicalise`. `P(ground)` is exactly invariant under a
64-element group (`beta_i += pi` per layer, and the global sign flip `(gamma, beta) ->
(-gamma, -beta)`), so every optimum has 63 duplicates. Folding into the fundamental domain
is what makes "did these two starts reach the same optimum?" well-posed, and it shrinks the
region the proposer has to cover by 64x.

## Search baselines (notebooks)

No model — pure optimisation over the angles, used as reference scores and as label generators.
Self-contained and GPU-first, meant for Kaggle/Colab; see `RUNNING.md`.

- `notebooks/02_direct_optimization_baseline.ipynb` — Adam from 32 random restarts per instance.
- `notebooks/03_massive_multistart.ipynb` — screen ~65k Sobol starting points per instance with a
  cheap forward-only pass, Adam-refine the survivors through a screen → coarse → fine funnel, and
  transfer elite angles found on one instance to the rest. Buys far more breadth per unit compute
  than restarts do, and §4 of the notebook measures that against notebook 02 instead of assuming
  it.
- `notebooks/04_cma_es_restarts.ipynb` — **IPOP-CMA-ES with restarts**, batched so that
  `runs × 500` independent evolution strategies advance in lockstep on the GPU. Where notebook 03
  buys breadth by sampling, this buys it by *adapting*: each run learns a covariance and a step
  size, so it explores along the landscape's own geometry rather than along the gradient. Converged
  runs are detected with Hansen's termination criteria and their GPU slots recycled in place —
  some reseeded from angles that won on other instances — and each wave doubles the population.
  §3 validates the implementation on sphere/ellipsoid/Rosenbrock/Rastrigin before it touches QAOA;
  §7 prices it against notebooks 02 and 03 at a matched evaluation budget.

  The interesting result is that whether a derivative-free method pays off here depends on the
  budget: below ~11.5k evaluations per instance Adam multistart wins, above ~24k CMA-ES wins.
  Since the two searches explore differently, their winning angles tend to sit in *different*
  basins, so the union of `cma_angles.npz` and `multistart_angles.npz` is better supervision — and
  a tighter oracle — than either alone.

## Running with Docker

Requires Docker with the NVIDIA container toolkit for GPU (a CPU fallback is provided).
Training the node-chain policy is a GPU-box job, and this is how it is run.

```bash
docker compose build                        # once, and after any change to src/

# ~1 minute at toy size, before committing to an 8h run.
# -cpu proves the image, the mounts and the code; the GPU one proves CUDA as well.
docker compose run --rm reinforce-smoke-cpu
docker compose run --rm reinforce-smoke

# the real training run, detached; checkpoints and history land in ./runs/reinforce
docker compose up -d reinforce
docker compose logs -f reinforce

# every hyperparameter is a flag — override by replacing the command
docker compose run --rm reinforce python -m src.reinforce --lr 1e-4 --chains 16 --iters 6

# inference -> runs/submission.csv, from the best checkpoint
docker compose run --rm reinforce-predict
# before h_test exists, measure the identical path on h_train:
docker compose run --rm reinforce-predict python -m src.reinforce \
    --ckpt runs/reinforce/best.pt --predict data/raw/h_train.npy \
    --submission runs/submission_train.csv

# no model: the gamma-box sweep, then the winning rung over all 500 instances (~10 min).
# Writes a submission — the floor under any learned result.
docker compose run --rm gscan
```

`./data` and `./runs` are bind-mounted, so `h_test.npy` is usable the day it lands with no
rebuild, and checkpoints survive the container. Every service is the same image
(`sberchall:latest`) with a different command, so the build happens once rather than once
per service.

A GPU service whose host has no `nvidia` container runtime fails at container start rather
than falling back to CPU — that is Docker's behaviour, not a bug here. On such a machine
`reinforce-smoke-cpu` and `train-cpu` are the ones that run.

The previous solution stays runnable:

```bash
# the entry point Kaggle runs: preflight -> the mode in src/experiment.py -> summary
docker compose run --rm experiment

# the learned-optimizer transformer: train / validate / predict
docker compose run --rm train
docker compose run --rm train-cpu        # no GPU available
docker compose run --rm validate
docker compose run --rm predict
```

Without Docker: `pip install -r requirements.txt`, then from the repo root

```bash
python -m src                 # the whole configured experiment (what Kaggle runs)
python -m src --quick         # same path, ~5 minutes, for checking a change works
```

or drive the stages individually — `python -m src.train`, `python -m src.validate`,
`python -m src.predict`. All hyperparameters are CLI flags with the intended defaults
(`python -m src.train --help`); `src/experiment.py`'s `CONFIG` is what fills them in for a
full run, so that dict is the thing to edit for a new experiment.

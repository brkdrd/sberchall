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

# Solution: learned-optimizer transformer

A decoder-only transformer conditioned on the instance vector `h` via **adaLN-Zero**
(DiT-style) that acts as a *learned optimiser* over QAOA angles. There are no angle
labels anywhere — the model is trained by backpropagating **through the differentiable
QAOA simulator** (`src/qaoa_ref.py`).

## How it works

Each sequence is an optimisation trajectory. A token is a 21-dim vector describing one
visited point in angle space:

```
[ 10 angles (5 gamma + 5 beta) | log P(ground) | d log P / d angles (10) ]
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

Code layout: `src/model.py` (transformer), `src/rollout.py` (sim-in-the-loop rollout),
`src/train.py`, `src/predict.py`, `src/qaoa_ref.py` (differentiable simulator).

## Running with Docker

Requires Docker with the NVIDIA container toolkit for GPU (a CPU fallback is provided).

```bash
# train (one command; checkpoints + logs land in ./runs)
docker compose run --rm train

# no GPU available:
docker compose run --rm train-cpu

# inference -> runs/submission.csv (expects data/raw/h_test.npy and runs/best.pt)
docker compose run --rm predict
# or with options:
docker compose run --rm predict python -m src.predict --h data/raw/h_test.npy --restarts 64 --polish 50
```

Without Docker: `pip install -r requirements.txt`, then `python -m src.train` and
`python -m src.predict` from the repo root. All hyperparameters are CLI flags with the
intended defaults (`python -m src.train --help`).

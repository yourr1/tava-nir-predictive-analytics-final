# Автоматизированная система предиктивной аналитики

НИР Попова Рината Васильевна, группа 5140201/50302.

Исследовательский фокус — допуск внешних табличных значений при неполной информации о времени доступности.

## Состав

- `src/` — программная реализация метода и экспериментов;
- `run_all.py` — запуск экспериментов;
- `verify_results.py` — проверка результатов;
- `tests/` — автоматические тесты;
- `TAVA_NIR_REPORT.pdf` — итоговый отчёт.
- `research_gap_experiment/` — синтетическая проверка нового исследовательского фокуса.

## Запуск

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe run_all.py --out rerun_results
.venv\Scripts\python.exe verify_results.py --results rerun_results
```

Для нового эксперимента:

```powershell
python -m pip install -r requirements.txt
python research_gap_experiment/run_uncertainty.py --out research_gap_experiment/results
python research_gap_experiment/verify_uncertainty.py --results research_gap_experiment/results
```

Полные входные данные и контрольные результаты находятся в итоговом архиве проекта.

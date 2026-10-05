# Battery SoH Cross-Dataset Project

State-of-Health (SoH) modeling across three battery datasets:

- NASA
- Oxford
- Warwick

## Repository Structure

Data (section 3.1):
- `nasa_explorer.py`, `oxford_explorer.py`, `warwick_explorer.py` - look inside the raw files (3.1.1)
- `data_teleport.py` - one-time copy of Warwick csv files into `data/WARWICK_RAW`
- `nasa_extractor.py`, `oxford_extractor.py`, `warwick_extractor.py` - raw files -> one row per discharge (3.1.2, 3.1.3)
- `data_merger.py` - joins the three tables into `output/MASTER_processed.csv`

Validation and Random Forest (sections 3.2-3.3):
- `split_comparison.py` - split variants A-D (3.2)
- `model_pipeline.py` + `main.py` - RF NASA + Oxford -> Warwick, SHAP plot (3.3, 3.6)
- `cross_val.py` - RF NASA + Warwick -> Oxford, actual vs predicted plot (3.3)
- `rf_feature_grid.py` - 3 directions x 4 feature sets (3.3)
- `common.py` - shared helpers

LSTM (sections 3.4-3.5):
- `lstm_cross_val.py` - LSTM with pre-training and fine-tuning on 2 Oxford cells
- `model_comparison.py` - LSTM vs RF vs cycle-only on the same windows, seeds and cell pairs
- `plot_pairs.py` - box plot for the 28 Oxford cell pairs

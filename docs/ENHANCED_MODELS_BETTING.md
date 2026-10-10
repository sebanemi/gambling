# Enhanced Models and Betting Strategy Design

## Overview

This document outlines the design for:
1. Enhanced prediction models (gradient boosting, hyperparameter optimization)
2. Betting strategy simulation (odds integration, bankroll management, backtesting)

## Phase 1: Enhanced Prediction Models

### 1.1 Gradient Boosting Models

**Motivation**: Gradient boosting models (XGBoost, LightGBM, CatBoost) often outperform logistic regression on tabular data with non-linear relationships.

**Implementation Plan**:

#### New Model: `GBMModel` (Gradient Boosting Machine)

Location: `src/football_predictor/models/gbm.py`

**Architecture**:
- Support multiple backends: XGBoost, LightGBM, CatBoost
- Configurable via settings: `MODELS_GBM_BACKEND` (default: "lightgbm")
- Hyperparameters configurable via settings or optimization
- Follows the same `Predictor` protocol as existing models
- Uses the same 128 features from `FeatureVector.flatten()`

**Hyperparameters (settings)**:
```python
models_gbm_backend: str = "lightgbm"  # xgboost, lightgbm, catboost
models_gbm_n_estimators: int = 100
models_gbm_max_depth: int = 6
models_gbm_learning_rate: float = 0.1
models_gbm_random_state: int = 42
```

**Key Features**:
- Same interface as `MLModel` (fit, predict)
- Automatic handling of missing values (if any)
- Feature importance extraction
- Early stopping support (if sufficient data)

**Integration**:
- Add to `ModelTrainer.train_all()` with key `"gbm"`
- Add to ensemble as optional member
- Add to CLI `--model` options

### 1.2 Hyperparameter Optimization Framework

**Motivation**: Automated hyperparameter tuning to find optimal model configurations.

**Implementation Plan**:

#### New Module: `src/football_predictor/models/hyperopt.py`

**Architecture**:
- Optuna-based Bayesian optimization
- Objective function: minimize log_loss on validation set
- Cross-validation: time-series split (respecting temporal order)
- Pruning: median pruner for early stopping of bad trials

**Settings**:
```python
models_hyperopt_n_trials: int = 50
models_hyperopt_timeout: int = 3600  # seconds
models_hyperopt_cv_folds: int = 5
```

**Integration**:
- New CLI command: `predictor optimize-hyperparams --model gbm`
- Updates settings with best parameters found
- Saves optimization history to `data/hyperopt/`

## Phase 2: Betting Strategy Simulation

### 2.1 Odds Integration

**Motivation**: To simulate betting, we need historical odds data.

**Database Schema Changes**:

#### New Table: `match_odds`

```sql
CREATE TABLE match_odds (
    id SERIAL PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    bookmaker VARCHAR(50) NOT NULL,
    home_win_odds DECIMAL(10, 4) NOT NULL,
    draw_odds DECIMAL(10, 4) NOT NULL,
    away_win_odds DECIMAL(10, 4) NOT NULL,
    odds_type VARCHAR(20) DEFAULT 'decimal',  -- decimal, fractional, american
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(match_id, bookmaker),
    INDEX idx_match_odds_match_id (match_id)
);
```

**Location**: `src/football_predictor/database/models/match_odds.py`

**Features**:
- Store odds from multiple bookmakers
- Support different odds formats (decimal default)
- Timestamp for odds capture time

**Data Sources**:
- Add odds providers to existing data providers
- Manual CSV import option
- API integration with odds providers (optional)

### 2.2 Value Bet Detection

**Motivation**: Identify bets where our model's probability implies positive expected value.

**New Module**: `src/football_predictor/betting/value.py`

**Algorithm**:
```python
def expected_value(model_prob: float, odds: float) -> float:
    """Calculate expected value of a bet."""
    return (model_prob * (odds - 1)) - ((1 - model_prob) * 1)

def is_value_bet(model_prob: float, odds: float, threshold: float = 0.05) -> bool:
    """Check if bet has positive expected value above threshold."""
    return expected_value(model_prob, odds) > threshold
```

**Settings**:
```python
betting_value_threshold: float = 0.05  # 5% edge
betting_min_odds: float = 1.1  # minimum odds to consider
betting_max_odds: float = 10.0  # maximum odds to consider
```

### 2.3 Bankroll Management Strategies

**New Module**: `src/football_predictor/betting/staking.py`

**Supported Strategies**:

1. **Fixed Stake**: Always bet the same amount
   ```python
   def fixed_stake(bankroll: float, stake: float) -> float:
       return stake
   ```

2. **Kelly Criterion**: Optimal fraction of bankroll
   ```python
   def kelly_stake(bankroll: float, model_prob: float, odds: float, kelly_fraction: float = 0.25) -> float:
       """Kelly criterion with fractional Kelly for safety."""
       edge = (model_prob * odds) - 1
       if edge <= 0:
           return 0.0
       kelly = edge / (odds - 1)
       return bankroll * kelly * kelly_fraction
   ```

3. **Proportional Stake**: Percentage of bankroll
   ```python
   def proportional_stake(bankroll: float, percentage: float) -> float:
       return bankroll * percentage
   ```

4. **Confidence-based**: Stake proportional to model confidence
   ```python
   def confidence_stake(bankroll: float, model_prob: float, odds: float, max_stake_pct: float = 0.05) -> float:
       ev = expected_value(model_prob, odds)
       normalized_ev = min(ev / 0.10, 1.0)  # cap at 10% EV
       return bankroll * max_stake_pct * normalized_ev
   ```

**Settings**:
```python
betting_strategy: str = "kelly"  # fixed, kelly, proportional, confidence
betting_initial_bankroll: float = 1000.0
betting_kelly_fraction: float = 0.25  # fractional Kelly for safety
betting_fixed_stake: float = 10.0
betting_proportional_percentage: float = 0.02  # 2% of bankroll
betting_max_stake_percentage: float = 0.05  # 5% max per bet
```

### 2.4 Betting Simulation Backtesting

**New Module**: `src/football_predictor/betting/backtester.py`

**Architecture**:
- Walk-forward simulation respecting temporal order
- For each match in historical order:
  1. Train models on past data only
  2. Generate predictions
  3. Get historical odds (if available)
  4. Calculate value bets
  5. Apply staking strategy
  6. Simulate bet outcome
  7. Update bankroll

**Metrics**:
- Total bets placed
- Win rate
- Return on Investment (ROI)
- Total profit/loss
- Maximum drawdown
- Sharpe ratio (risk-adjusted returns)
- Average stake size

**Output**:
```python
@dataclass
class BettingSimulationResult:
    total_bets: int
    winning_bets: int
    losing_bets: int
    win_rate: float
    total_staked: float
    total_return: float
    roi: float
    max_drawdown: float
    final_bankroll: float
    bankroll_history: list[tuple[date, float]]
```

**Settings**:
```python
betting_backtest_min_matches: int = 10  # minimum history before betting
betting_backtest_start_date: date | None = None  # optional start date
```

### 2.5 CLI Integration

**New Commands**:

1. `predictor import-odds` - Import odds data
   ```bash
   predictor import-odds --path data/odds.csv
   predictor import-odds --provider odds-api --competition PL --season 2024
   ```

2. `predictor simulate-betting` - Run betting simulation
   ```bash
   predictor simulate-betting --model ensemble --strategy kelly --initial-bankroll 1000
   predictor simulate-betting --model gbm --strategy confidence --start-date 2024-01-01
   ```

3. `predictor value-bets` - Find current value bets
   ```bash
   predictor value-bets --model ensemble --threshold 0.05
   ```

### 2.6 Database Schema for Betting Results

#### New Table: `betting_simulations`

```sql
CREATE TABLE betting_simulations (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    initial_bankroll DECIMAL(15, 2) NOT NULL,
    final_bankroll DECIMAL(15, 2) NOT NULL,
    total_bets INTEGER NOT NULL,
    winning_bets INTEGER NOT NULL,
    roi DECIMAL(10, 4) NOT NULL,
    max_drawdown DECIMAL(10, 4) NOT NULL,
    parameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

#### New Table: `betting_history`

```sql
CREATE TABLE betting_history (
    id SERIAL PRIMARY KEY,
    simulation_id INTEGER NOT NULL REFERENCES betting_simulations(id) ON DELETE CASCADE,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    prediction VARCHAR(50) NOT NULL,  -- HOME, DRAW, AWAY
    model_probability DECIMAL(10, 4) NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    odds DECIMAL(10, 4) NOT NULL,
    stake DECIMAL(15, 2) NOT NULL,
    profit_loss DECIMAL(15, 2) NOT NULL,
    bankroll_after DECIMAL(15, 2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_betting_history_simulation_id (simulation_id),
    INDEX idx_betting_history_match_id (match_id)
);
```

## Implementation Order

1. **Enhanced Models**
   - [ ] Add XGBoost/LightGBM/CatBoost dependencies to pyproject.toml
   - [ ] Implement `GBMModel` class
   - [ ] Integrate into `ModelTrainer`
   - [ ] Add to CLI
   - [ ] Write tests
   - [ ] Implement hyperparameter optimization
   - [ ] Add hyperopt CLI command

2. **Betting Infrastructure**
   - [ ] Create `match_odds` table and migration
   - [ ] Implement odds import (CSV provider)
   - [ ] Implement value bet detection
   - [ ] Implement staking strategies
   - [ ] Create betting backtester
   - [ ] Add betting simulation tables and migrations
   - [ ] Integrate into CLI
   - [ ] Write tests

3. **Documentation**
   - [ ] Update README with new features
   - [ ] Update MODELOS.md with GBM details
   - [ ] Create BETTING.md with betting guide
   - [ ] Update ROADMAP.md

## Dependencies to Add

```toml
[project.optional-dependencies]
ml-extra = [
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "catboost>=1.2.0",
    "optuna>=3.0.0",
]
```

## Notes

- All betting simulations must respect anti-leakage principles
- Odds data should be captured before match completion
- Bankroll management should include safety limits (max drawdown, max stake)
- Hyperparameter optimization should use time-series cross-validation
- All new features should be tested

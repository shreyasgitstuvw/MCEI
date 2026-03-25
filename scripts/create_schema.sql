-- ============================================================
-- NSE Analytics Platform - Database Schema
-- Run with: psql -U nse_user -d nse_analytics -f scripts/create_schema.sql
-- ============================================================

-- Daily price data (one row per symbol per day)
CREATE TABLE IF NOT EXISTS fact_daily_prices (
    id                  BIGSERIAL PRIMARY KEY,
    trade_date          DATE          NOT NULL,
    symbol              VARCHAR(50),
    series              VARCHAR(10),
    security_name       VARCHAR(255)  NOT NULL,
    isin                VARCHAR(20),
    market              VARCHAR(10),
    prev_close          DECIMAL(15,2),
    open_price          DECIMAL(15,2),
    high_price          DECIMAL(15,2),
    low_price           DECIMAL(15,2),
    close_price         DECIMAL(15,2),
    net_traded_value    DECIMAL(20,2),
    net_traded_qty      BIGINT,
    total_trades        INTEGER,
    delivery_qty        BIGINT,
    delivery_pct        DECIMAL(10,4),
    high_52_week        DECIMAL(15,2),
    low_52_week         DECIMAL(15,2),
    day_change          DECIMAL(15,4),
    day_change_pct      DECIMAL(10,4),
    intraday_range      DECIMAL(15,4),
    intraday_range_pct  DECIMAL(10,4),
    is_index            BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_price_symbol_date UNIQUE (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_prices_date       ON fact_daily_prices(trade_date);
CREATE INDEX IF NOT EXISTS idx_prices_symbol      ON fact_daily_prices(symbol);
CREATE INDEX IF NOT EXISTS idx_prices_date_symbol ON fact_daily_prices(trade_date, symbol);

-- Circuit breaker hits
CREATE TABLE IF NOT EXISTS fact_circuit_hits (
    id            BIGSERIAL PRIMARY KEY,
    trade_date    DATE         NOT NULL,
    symbol        VARCHAR(50)  NOT NULL,
    series        VARCHAR(10),
    security_name VARCHAR(255),
    circuit_type  CHAR(1),             -- 'H' upper / 'L' lower
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_circuit_symbol_date UNIQUE (symbol, trade_date, circuit_type)
);

CREATE INDEX IF NOT EXISTS idx_circuit_date   ON fact_circuit_hits(trade_date);
CREATE INDEX IF NOT EXISTS idx_circuit_symbol ON fact_circuit_hits(symbol);

-- Corporate actions
CREATE TABLE IF NOT EXISTS fact_corporate_actions (
    id            BIGSERIAL PRIMARY KEY,
    trade_date    DATE         NOT NULL,
    symbol        VARCHAR(50),
    series        VARCHAR(10),
    security_name VARCHAR(255),
    record_date   DATE,
    ex_date       DATE,
    purpose       TEXT,
    action_type   VARCHAR(50),          -- Dividend / Bonus / Split / Rights etc.
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ca_date    ON fact_corporate_actions(trade_date);
CREATE INDEX IF NOT EXISTS idx_ca_symbol  ON fact_corporate_actions(symbol);
CREATE INDEX IF NOT EXISTS idx_ca_ex_date ON fact_corporate_actions(ex_date);

-- Market capitalisation
CREATE TABLE IF NOT EXISTS fact_market_cap (
    id            BIGSERIAL PRIMARY KEY,
    trade_date    DATE          NOT NULL,
    symbol        VARCHAR(50)   NOT NULL,
    series        VARCHAR(10),
    security_name VARCHAR(255),
    face_value    DECIMAL(10,2),
    issue_size    BIGINT,
    close_price   DECIMAL(15,2),
    market_cap    DECIMAL(25,2),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_mcap_symbol_date UNIQUE (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_mcap_date   ON fact_market_cap(trade_date);
CREATE INDEX IF NOT EXISTS idx_mcap_symbol ON fact_market_cap(symbol);

-- 52-week highs/lows
CREATE TABLE IF NOT EXISTS fact_hl_hits (
    id            BIGSERIAL PRIMARY KEY,
    trade_date    DATE         NOT NULL,
    symbol        VARCHAR(50)  NOT NULL,
    series        VARCHAR(10),
    security_name VARCHAR(255),
    hl_type       CHAR(1),             -- 'H' new high / 'L' new low
    price         DECIMAL(15,2),       -- LTP at the breakout
    prev_high_low DECIMAL(15,2),       -- previous 52w level crossed
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hl_date   ON fact_hl_hits(trade_date);
CREATE INDEX IF NOT EXISTS idx_hl_symbol ON fact_hl_hits(symbol);

-- ETF daily data
CREATE TABLE IF NOT EXISTS fact_etf_prices (
    id                BIGSERIAL PRIMARY KEY,
    trade_date        DATE          NOT NULL,
    symbol            VARCHAR(50)   NOT NULL,
    security_name     VARCHAR(255),
    underlying        VARCHAR(255),
    prev_close        DECIMAL(15,2),
    open_price        DECIMAL(15,2),
    high_price        DECIMAL(15,2),
    low_price         DECIMAL(15,2),
    close_price       DECIMAL(15,2),
    net_traded_value  DECIMAL(20,2),
    net_traded_qty    BIGINT,
    total_trades      INTEGER,
    high_52_week      DECIMAL(15,2),
    low_52_week       DECIMAL(15,2),
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_etf_symbol_date UNIQUE (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_etf_date   ON fact_etf_prices(trade_date);
CREATE INDEX IF NOT EXISTS idx_etf_symbol ON fact_etf_prices(symbol);

-- Top traded securities
CREATE TABLE IF NOT EXISTS fact_top_traded (
    id                BIGSERIAL PRIMARY KEY,
    trade_date        DATE          NOT NULL,
    rank              INTEGER,
    security_name     VARCHAR(255),
    net_traded_value  DECIMAL(20,2),
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tt_date ON fact_top_traded(trade_date);

-- ETL ingestion log
CREATE TABLE IF NOT EXISTS log_ingestion (
    id                    BIGSERIAL PRIMARY KEY,
    trade_date            DATE          NOT NULL,
    file_name             VARCHAR(255),
    status                VARCHAR(50),  -- SUCCESS / FAILED / PARTIAL
    records_processed     INTEGER,
    error_message         TEXT,
    processing_seconds    INTEGER,
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_log_date ON log_ingestion(trade_date);

-- Useful view: today's movers
CREATE OR REPLACE VIEW v_daily_movers AS
SELECT
    symbol,
    security_name,
    close_price,
    prev_close,
    ROUND(((close_price - prev_close) / NULLIF(prev_close, 0)) * 100, 2) AS pct_change,
    net_traded_value,
    net_traded_qty,
    total_trades,
    trade_date
FROM fact_daily_prices
WHERE is_index = FALSE
ORDER BY pct_change DESC;

-- Useful view: market breadth per day
CREATE OR REPLACE VIEW v_market_breadth AS
SELECT
    trade_date,
    COUNT(*) FILTER (WHERE close_price > prev_close) AS advances,
    COUNT(*) FILTER (WHERE close_price < prev_close) AS declines,
    COUNT(*) FILTER (WHERE close_price = prev_close) AS unchanged,
    COUNT(*)                                          AS total,
    SUM(net_traded_value)                             AS total_turnover
FROM fact_daily_prices
WHERE is_index = FALSE
GROUP BY trade_date
ORDER BY trade_date DESC;
-- ============================================================
-- Pre-computed analytics tables (populated by scripts/precompute_analytics.py)
-- ============================================================

CREATE TABLE IF NOT EXISTS precomp_regime (
    id                SERIAL PRIMARY KEY,
    computed_date     DATE    NOT NULL,
    trade_date        DATE    NOT NULL,
    close_price       NUMERIC,
    trend             TEXT,
    volatility_regime TEXT,
    market_regime     TEXT,
    adx               NUMERIC,
    rsi               NUMERIC,
    macd_hist         NUMERIC,
    atr_pct           NUMERIC,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (computed_date, trade_date)
);

CREATE TABLE IF NOT EXISTS precomp_volume_patterns (
    id                  SERIAL PRIMARY KEY,
    computed_date       DATE    NOT NULL,
    pattern_type        TEXT    NOT NULL,
    symbol              TEXT    NOT NULL,
    trade_date          DATE    NOT NULL,
    net_traded_qty      BIGINT,
    volume_ma           NUMERIC,
    breakout_magnitude  NUMERIC,
    price_change        NUMERIC,
    dryup_streak        INTEGER,
    price_range_pct     NUMERIC,
    price_direction     TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS precomp_causality (
    id            SERIAL PRIMARY KEY,
    computed_date DATE    NOT NULL,
    symbol        TEXT    NOT NULL,
    lag1_corr     NUMERIC,
    direction     TEXT,
    strength      TEXT,
    avg_corr      NUMERIC,
    n_dates       INTEGER,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (computed_date, symbol)
);

-- Indexes on hot query columns
CREATE INDEX IF NOT EXISTS idx_daily_prices_date        ON fact_daily_prices (trade_date);
CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol      ON fact_daily_prices (symbol);
CREATE INDEX IF NOT EXISTS idx_daily_prices_date_symbol ON fact_daily_prices (trade_date, symbol);
CREATE INDEX IF NOT EXISTS idx_daily_prices_series      ON fact_daily_prices (trade_date, series, is_index);
CREATE INDEX IF NOT EXISTS idx_daily_prices_delivery    ON fact_daily_prices (trade_date, series) WHERE delivery_pct IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_hl_type                  ON fact_hl_hits (hl_type);
CREATE INDEX IF NOT EXISTS idx_precomp_regime_date      ON precomp_regime (computed_date, trade_date);
CREATE INDEX IF NOT EXISTS idx_precomp_vol_date         ON precomp_volume_patterns (computed_date, pattern_type);
CREATE INDEX IF NOT EXISTS idx_precomp_caus_date        ON precomp_causality (computed_date);

-- ============================================================
-- Migration: add columns to existing tables (safe to re-run)
-- ============================================================
ALTER TABLE fact_daily_prices ADD COLUMN IF NOT EXISTS isin               VARCHAR(20);
ALTER TABLE fact_daily_prices ADD COLUMN IF NOT EXISTS delivery_qty       BIGINT;
ALTER TABLE fact_daily_prices ADD COLUMN IF NOT EXISTS delivery_pct       DECIMAL(10,4);
ALTER TABLE fact_daily_prices ADD COLUMN IF NOT EXISTS day_change         DECIMAL(15,4);
ALTER TABLE fact_daily_prices ADD COLUMN IF NOT EXISTS day_change_pct     DECIMAL(10,4);
ALTER TABLE fact_daily_prices ADD COLUMN IF NOT EXISTS intraday_range     DECIMAL(15,4);
ALTER TABLE fact_daily_prices ADD COLUMN IF NOT EXISTS intraday_range_pct DECIMAL(10,4);

ALTER TABLE fact_hl_hits ADD COLUMN IF NOT EXISTS price         DECIMAL(15,2);
ALTER TABLE fact_hl_hits ADD COLUMN IF NOT EXISTS prev_high_low DECIMAL(15,2);

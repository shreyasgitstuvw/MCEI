-- ============================================================
-- NSE Analytics Platform - Database Setup
-- Run as PostgreSQL superuser (postgres)
-- ============================================================

-- Grant permissions to nse_user on public schema
GRANT ALL ON SCHEMA public TO analyst;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO analyst;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO analyst;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO analyst;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO analyst;

-- Now create tables
-- Daily price data (one row per symbol per day)
CREATE TABLE IF NOT EXISTS fact_daily_prices (
    id               BIGSERIAL PRIMARY KEY,
    trade_date       DATE          NOT NULL,
    symbol           VARCHAR(50),
    series           VARCHAR(10),
    security_name    VARCHAR(255)  NOT NULL,
    isin             VARCHAR(20),
    market           VARCHAR(10),
    prev_close       DECIMAL(15,2),
    open_price       DECIMAL(15,2),
    high_price       DECIMAL(15,2),
    low_price        DECIMAL(15,2),
    close_price      DECIMAL(15,2),
    net_traded_value DECIMAL(20,2),
    net_traded_qty   BIGINT,
    total_trades     INTEGER,
    high_52_week     DECIMAL(15,2),
    low_52_week      DECIMAL(15,2),
    is_index         BOOLEAN DEFAULT FALSE,
    is_valid         BOOLEAN DEFAULT TRUE,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
    isin              VARCHAR(20),
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

-- Grant permissions on newly created objects
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO nse_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO nse_user;

-- Confirm setup
SELECT 'Database setup complete!' AS status;
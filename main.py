from flask import Flask, request, jsonify
import json
import logging
import traceback
import re
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

# ─── Load .env before any os.getenv() calls ─────────────────────────────────
load_dotenv()
# ─── Load API credentials into globals ─────────────────────────────────────
ALPACA_API_KEY    = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# ─── Initialize Alpaca trading client ──────────────────────────────────────
api = tradeapi.REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
)

# ─── NEW: derive Data API base URL for HTTP bars ───────────────────────────
DATA_BASE_URL = os.getenv(
    'ALPACA_DATA_URL',
    os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        .replace('paper-api', 'data')
)

# ─── Logging configuration ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('webhook.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── Flask app setup ───────────────────────────────────────────────────────
app = Flask(__name__)

# ─── Webhook token for validation ──────────────────────────────────────────
WEBHOOK_TOKEN = os.getenv('WEBHOOK_TOKEN')
# ─── Trading configuration constants ───────────────────────────────────────
INITIAL_BALANCE_PER_TICKER = 2000   # $ per new ticker
BALANCE_USAGE_PERCENT    = 0.98     # use 98%
BALANCE_FILE             = 'ticker_balances.json'
TRADE_LOG_FILE           = 'trade_log.json'


def load_ticker_balances():
    """Load ticker balances from file or initialize empty"""
    try:
        with open(BALANCE_FILE, 'r') as f:
            balances = json.load(f)
        logger.info(f"Loaded ticker balances from file: {balances}")
        return balances
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.info(f"No balances file found ({e}), starting fresh")
        return {}

def save_ticker_balances(balances):
    """Save ticker balances to file"""
    try:
        with open(BALANCE_FILE, 'w') as f:
            json.dump(balances, f, indent=2)
        logger.info(f"Saved ticker balances: {balances}")
    except Exception as e:
        logger.error(f"Error saving ticker balances: {e}")

def save_trade_log(trade_log):
    """Save trade log to file"""
    try:
        with open(TRADE_LOG_FILE, 'w') as f:
            json.dump(trade_log, f, indent=2)
        logger.info(f"Saved trade log")
    except Exception as e:
        logger.error(f"Error saving trade log: {e}")
def load_trade_log():
    """Load trade log from file or initialize empty"""
    try:
        with open(TRADE_LOG_FILE, 'r') as f:
            log = json.load(f)
        logger.info("Loaded trade log from file")
        return log
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.info(f"No trade log file found ({e}), starting fresh")
        return {}


def initialize_ticker_balance(symbol, balances):
    """Initialize a new ticker with starting balance"""
    if symbol not in balances:
        balances[symbol] = {
            'balance': INITIAL_BALANCE_PER_TICKER,
            'total_invested': 0.0,
            'total_realized_pnl': 0.0,
            'trade_count': 0
        }
        logger.info(f"NEW TICKER DETECTED: {symbol} - Allocated initial balance: ${INITIAL_BALANCE_PER_TICKER}")
        save_ticker_balances(balances)
    return balances[symbol]


def get_ticker_balance(symbol, balances):
    """Get ticker balance, initialize if new"""
    return initialize_ticker_balance(symbol, balances)


def calculate_buy_amount_and_shares(symbol, balance, current_price):
    """Calculate how much to invest (98% of balance) and how many shares to buy"""
    usable_amount = balance * BALANCE_USAGE_PERCENT
    shares_to_buy = usable_amount / current_price

    logger.info(f"BUY CALCULATION for {symbol}:")
    logger.info(f"  Current balance: ${balance:.2f}")
    logger.info(f"  Usable amount (98%): ${usable_amount:.2f}")
    logger.info(f"  Current price: ${current_price:.4f}")
    logger.info(f"  Shares to buy: {shares_to_buy:.6f}")

    return usable_amount, shares_to_buy


def log_buy_trade(symbol, shares, price, total_cost):
    """Log a buy trade"""
    trade_log = load_trade_log()

    # Clear previous log for this ticker (new buy alert)
    trade_log[symbol] = {
        'type': 'buy',
        'shares': shares,
        'price': price,
        'total_cost': total_cost,
        'timestamp': datetime.now().isoformat()
    }

    save_trade_log(trade_log)
    logger.info(f"LOGGED BUY: {symbol} - {shares:.6f} shares at ${price:.4f} = ${total_cost:.2f}")


def log_sell_trade(symbol, shares, price, total_proceeds):
    """Log a sell trade and calculate P&L"""
    trade_log = load_trade_log()

    if symbol not in trade_log:
        logger.error(f"No buy record found for {symbol} - cannot calculate P&L")
        return 0.0

    buy_record = trade_log[symbol]
    buy_cost = buy_record['total_cost']
    pnl = total_proceeds - buy_cost

    # Add sell information to existing record
    trade_log[symbol]['sell'] = {
        'shares': shares,
        'price': price,
        'total_proceeds': total_proceeds,
        'pnl': pnl,
        'timestamp': datetime.now().isoformat()
    }

    save_trade_log(trade_log)
    logger.info(f"LOGGED SELL: {symbol} - {shares:.6f} shares at ${price:.4f} = ${total_proceeds:.2f}")
    logger.info(f"P&L for {symbol}: ${pnl:.2f}")

    return pnl


def sync_position_with_alpaca(symbol, ticker_balances):
    """Sync internal tracking with actual Alpaca position"""
    try:
        # Get actual position from Alpaca
        try:
            alpaca_position = api.get_position(symbol)
            actual_shares = float(alpaca_position.qty)
            actual_avg_cost = float(alpaca_position.avg_entry_price)
            actual_market_value = float(alpaca_position.market_value)

            logger.info(f"ALPACA ACTUAL POSITION for {symbol}:")
            logger.info(f"  Shares: {actual_shares}")
            logger.info(f"  Avg Cost: ${actual_avg_cost:.4f}")
            logger.info(f"  Market Value: ${actual_market_value:.2f}")

            return actual_shares

        except tradeapi.rest.APIError as e:
            # No position exists in Alpaca
            logger.info(f"No actual position in Alpaca for {symbol}: {e}")
            return 0.0

    except Exception as e:
        logger.error(f"Error syncing position for {symbol}: {e}")
        return 0.0


def update_balance_after_buy(symbol, balances, investment_amount):
    """Update ticker balance after a buy order"""
    ticker_data = get_ticker_balance(symbol, balances)

    # Deduct investment from balance
    ticker_data['balance'] -= investment_amount
    ticker_data['total_invested'] += investment_amount
    ticker_data['trade_count'] += 1

    logger.info(f"BALANCE UPDATE AFTER BUY for {symbol}:")
    logger.info(f"  Investment amount: ${investment_amount:.2f}")
    logger.info(f"  New balance: ${ticker_data['balance']:.2f}")
    logger.info(f"  Total invested: ${ticker_data['total_invested']:.2f}")

    balances[symbol] = ticker_data
    save_ticker_balances(balances)


def update_balance_after_sell(symbol, balances, proceeds):
    """Update ticker balance after a sell order"""
    ticker_data = get_ticker_balance(symbol, balances)

    # Add proceeds to balance
    ticker_data['balance'] += proceeds

    # Calculate P&L from trade log
    trade_log = load_trade_log()
    if symbol in trade_log and 'sell' in trade_log[symbol]:
        pnl = trade_log[symbol]['sell']['pnl']
        ticker_data['total_realized_pnl'] += pnl

    logger.info(f"BALANCE UPDATE AFTER SELL for {symbol}:")
    logger.info(f"  Proceeds: ${proceeds:.2f}")
    logger.info(f"  New balance: ${ticker_data['balance']:.2f}")
    logger.info(f"  Total realized P&L: ${ticker_data['total_realized_pnl']:.2f}")

    balances[symbol] = ticker_data
    save_ticker_balances(balances)


def fix_json_format(json_string):
    """Fix the specific JSON issue: remove extra quotes after numbers"""
    try:
        original_string = json_string
        logger.info(f"Attempting to fix JSON format...")

        # Remove outer single quotes if present
        if json_string.startswith("'") and json_string.endswith("'"):
            json_string = json_string[1:-1]
            logger.info("Removed outer single quotes")

        # THE ACTUAL PROBLEM: Extra quotes after numbers
        # Pattern: number"comma -> number,
        # Pattern: number"} -> number}
        json_string = re.sub(r'(\d+\.?\d*)"([,}])', r'\1\2', json_string)
        logger.info("Removed extra quotes after numbers")

        if json_string != original_string:
            logger.info(f"JSON was modified during fix")
            logger.info(f"Original: {original_string}")
            logger.info(f"Fixed:    {json_string}")

        return json_string

    except Exception as e:
        logger.error(f"Error in fix_json_format: {e}")
        return original_string

def get_bid_ask(symbol):
    """Fetch the latest NBBO quote for symbol and return (bid, ask)."""
    url = f"{DATA_BASE_URL}/v2/stocks/{symbol}/quotes/latest"
    headers = {
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    r = requests.get(url, headers=headers, timeout=5)
    r.raise_for_status()
    q = r.json().get("quote", {})
    bid = float(q.get("bp", 0))
    ask = float(q.get("ap", 0))
    return bid, ask

def extract_essential_data(data):
    """Extract only the essential fields needed for trading"""
    essential_fields = {
        'symbol': data.get('symbol'),
        'action': data.get('action'),
        'token': data.get('token')
    }

    # Only include price if it exists and is valid
    if 'price' in data and data['price'] is not None:
        try:
            # Ensure price is a valid number
            price_val = float(data['price'])
            essential_fields['price'] = price_val
        except (ValueError, TypeError):
            logger.warning(f"Invalid price value ignored: {data['price']}")

    logger.info(f"Extracted essential data: {json.dumps(essential_fields, indent=2)}")
    return essential_fields


def get_current_price(symbol):
    """Fetch the very last trade (incl. extended hours) via HTTP, else quote midpoint."""
    headers = {
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }

    # 1️⃣ Try the latest trade endpoint (always returns the newest print)
    try:
        url = f"{DATA_BASE_URL}/v2/stocks/{symbol}/trades/latest"
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json().get("trade", {})
        if data and "p" in data:
            price = float(data["p"])
            logger.info(f"Latest trade/latest price for {symbol}: ${price}")
            return price
    except Exception as e:
        logger.warning(f"/trades/latest failed: {e}")

    # 2️⃣ Fallback to the latest quote midpoint
    try:
        url = f"{DATA_BASE_URL}/v2/stocks/{symbol}/quotes/latest"
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        q = r.json().get("quote", {})
        if q and "bp" in q and "ap" in q:
            bid = float(q["bp"])
            ask = float(q["ap"])
            mid = (bid + ask) / 2
            logger.info(f"Quote midpoint for {symbol}: ${mid}")
            return mid
    except Exception as e:
        logger.error(f"/quotes/latest failed: {e}")

    logger.error(f"Could not fetch a live price for {symbol}")
    return None

def cancel_pending_orders(symbol):
    """Cancel all pending orders for a symbol"""
    try:
        orders = api.list_orders(status='open', symbols=[symbol])
        cancelled_count = 0
        for order in orders:
            try:
                api.cancel_order(order.id)
                logger.info(
                    f"Cancelled pending order: {order.id} ({order.side} {order.qty} {order.symbol} @ ${order.limit_price})")
                cancelled_count += 1
            except Exception as e:
                logger.warning(f"Failed to cancel order {order.id}: {e}")

        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} pending orders for {symbol}")
        else:
            logger.info(f"No pending orders to cancel for {symbol}")

    except Exception as e:
        logger.error(f"Error cancelling orders for {symbol}: {e}")


@app.route('/webhook', methods=['POST'])

def webhook():
    try:
        logger.info("NEW WEBHOOK ALERT RECEIVED")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")

        # Load current ticker balances
        ticker_balances = load_ticker_balances()

        # Get raw data
        raw_data = request.get_data()
        logger.info(f"Raw request data: {raw_data}")
        logger.info(f"Raw data type: {type(raw_data)}")
        logger.info(f"Content-Type: {request.content_type}")

        # Try extracting only essential fields without full parsing
        try:
            decoded_data = raw_data.decode('utf-8')
            logger.info(f"Decoded raw alert text:\n{decoded_data}")

            # Use pattern matching to pull out only the required fields
            def extract_required_fields(raw_text):
                fields = {}
                patterns = {
                    "symbol": r'"symbol"\s*:\s*"([^"]+)"',
                    "action": r'"action"\s*:\s*"([^"]+)"',
                    "price": r'"price"\s*:\s*([0-9.]+)',
                    "token": r'"token"\s*:\s*"([^"]+)"'
                }

                for key, pattern in patterns.items():
                    match = re.search(pattern, raw_text)
                    if match:
                        value = match.group(1)
                        if key == "price":
                            try:
                                value = float(value)
                            except ValueError:
                                logger.warning(f"Non-numeric value for {key}: {value}")
                                continue
                        fields[key] = value
                    else:
                        if key != "price":  # Price is optional
                            logger.warning(f"Missing expected field: {key}")

                # Validate core fields
                for key in ['symbol', 'action', 'token']:
                    if key not in fields:
                        raise ValueError(f"Missing required field: {key}")

                return fields

            data = extract_required_fields(decoded_data)
            safe_data = data.copy()
            if "token" in safe_data:
                safe_data["token"] = "***REDACTED***"
            logger.info(f"Clean extracted data: {json.dumps(safe_data, indent=2)}")

        except Exception as e:
            logger.error(f"Failed to extract required fields: {e}")
            return jsonify({'error': 'Invalid or malformed alert data'}), 400

        # Extract only essential data for processing
        data = extract_essential_data(data)

        # Validate token
        received_token = data.get('token')
        #logger.info(f"Token validation - Received: {received_token}")

        if received_token != WEBHOOK_TOKEN:
            logger.error("Invalid token!")
            return jsonify({'error': 'Invalid token'}), 401

        # Extract required fields
        symbol = data.get('symbol')
        action = data.get('action')
        price = data.get('price')  # Optional

        logger.info("Extracted essential fields:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Action: {action}")
        logger.info(f"   Price: {price}")

        # Validate required fields
        if not all([symbol, action]):
            logger.error("Missing required fields")
            return jsonify({'error': 'Missing required fields: symbol, action'}), 400

        # Convert action to side
        side_mapping = {
            'buy': 'buy',
            'sell': 'sell',
            'long': 'buy',
            'short': 'sell'
        }

        side = side_mapping.get(action.lower())
        if not side:
            logger.error(f"Invalid action: {action}")
            return jsonify({'error': f'Invalid action: {action}'}), 400

        logger.info(f"Action '{action}' mapped to side '{side}'")

        # SYNC POSITION WITH ALPACA
        logger.info("=== SYNCING POSITION WITH ALPACA ===")
        actual_shares = sync_position_with_alpaca(symbol, ticker_balances)

        # Get ticker balance data (initialize if new)
        ticker_data = get_ticker_balance(symbol, ticker_balances)
        current_balance = ticker_data['balance']

        logger.info(f"Current balance for {symbol}: ${current_balance:.2f}")
        logger.info(f"Position after sync: {actual_shares} shares")

        # Check if market is open
        market_clock = api.get_clock()
        is_market_open = market_clock.is_open
        logger.info(f"Market is currently: {'OPEN' if is_market_open else 'CLOSED'}")

        # ─── FIGURE OUT OUR PRICE BASED ON MARKET HOURS ──────────────────────
        market_clock = api.get_clock()
        is_market_open = market_clock.is_open

        if is_market_open:
            # RTH → fetch the *actual* live price
            current_price = get_current_price(symbol)
        else:
            # OTH → use the Pine price from your alert (if present)
            current_price = price or get_current_price(symbol)

        if current_price is None:
            logger.error("Could not retrieve current price")
            return jsonify({'error': 'Could not retrieve current price'}), 500

        # Cancel any existing pending orders for this symbol
        cancel_pending_orders(symbol)

        # ─── Handle SELL orders ────────────────────────────────────────────
        if side == 'sell':
            logger.info("=== SELL ORDER HANDLER ===")

            if actual_shares <= 0:
                logger.error("Cannot sell - no position exists")
                return jsonify({'error': 'No position to sell'}), 400

            qty = actual_shares

            if is_market_open:
                logger.info("Market is open → submitting MARKET sell order")
                try:
                    order = api.submit_order(
                        symbol=symbol.upper(),
                        qty=str(qty),
                        side='sell',
                        type='market',
                        time_in_force='day',
                        extended_hours=False
                    )
                    execution_price = current_price
                    proceeds = qty * execution_price
                except Exception as e:
                    logger.error(f"Failed to submit MARKET sell order: {e}")
                    return jsonify({'error': str(e)}), 500

            else:
                limit_price = round(float(price) * 0.998, 2)
                logger.info(f"Extended hours → LIMIT sell @ {price}×0.998 → ${limit_price}")
                try:
                    order = api.submit_order(
                        symbol=symbol.upper(),
                        qty=str(qty),
                        side='sell',
                        type='limit',
                        limit_price=str(limit_price),
                        time_in_force='day',
                        extended_hours=True,
                       )
                    execution_price = limit_price
                    proceeds = qty * execution_price
                except Exception as e:
                    logger.error(f"Failed to submit LIMIT sell order: {e}")
                    return jsonify({'error': str(e)}), 500

            logger.info(f"SUCCESS: Sell order submitted - ID: {order.id}")
            pnl = log_sell_trade(symbol, qty, execution_price, proceeds)
            update_balance_after_sell(symbol, ticker_balances, proceeds)

            # ─── RETURN from SELL ──────────────────────────────────────────
            return jsonify({
                'success': True,
                'order_id': order.id,
                'shares_sold': qty,
                'estimated_proceeds': f'${proceeds:.2f}',
                'pnl': f'${pnl:.2f}',
                'updated_balance': f'${ticker_balances[symbol]["balance"]:.2f}'
            }), 200

        # ─── Handle BUY orders ─────────────────────────────────────────────
        else:
            logger.info("=== BUY ORDER HANDLER ===")

            if actual_shares > 0:
                logger.error(f"Cannot buy - position already exists ({actual_shares} shares)")
                return jsonify({'error': 'Position already exists'}), 400

            if current_balance <= 0:
                logger.error(f"No available balance for {symbol}: ${current_balance}")
                return jsonify({'error': f'No available balance for {symbol}'}), 400

            investment_amount, qty = calculate_buy_amount_and_shares(
                symbol, current_balance, current_price
            )
            if qty <= 0:
                logger.error(f"Cannot buy any shares with ${current_balance}")
                return jsonify({'error': 'Insufficient balance'}), 400

            logger.info(f"CALCULATED: Invest ${investment_amount:.2f} → {qty:.6f} shares")

            if is_market_open:
                logger.info("Market is open → submitting MARKET buy order")
                try:
                    order = api.submit_order(
                        symbol=symbol.upper(),
                        qty=str(qty),
                        side='buy',
                        type='market',
                        time_in_force='day',
                        extended_hours=False
                    )
                    execution_price = current_price
                    actual_cost = investment_amount
                except Exception as e:
                    logger.error(f"Failed to submit MARKET buy order: {e}")
                    return jsonify({'error': str(e)}), 500

            else:
                limit_price = round(float(price) * 1.005, 2)
                logger.info(f"Extended hours → LIMIT buy @ {price}×1.005 → ${limit_price}")
                try:
                    order = api.submit_order(
                        symbol=symbol.upper(),
                        qty=str(qty),
                        side='buy',
                        type='limit',
                        limit_price=str(limit_price),
                        time_in_force='day',
                        extended_hours=True,
                    )
                    execution_price = limit_price
                    actual_cost = qty * execution_price
                except Exception as e:
                    logger.error(f"Failed to submit LIMIT buy order: {e}")
                    return jsonify({'error': str(e)}), 500

            logger.info(f"SUCCESS: Buy order submitted - ID: {order.id}")
            log_buy_trade(symbol, qty, execution_price, actual_cost)
            update_balance_after_buy(symbol, ticker_balances, actual_cost)

            # ─── RETURN from BUY ────────────────────────────────────────────
            return jsonify({
                'success': True,
                'order_id': order.id,
                'shares_bought': qty,
                'estimated_cost': f'${actual_cost:.2f}',
                'remaining_balance': f'${ticker_balances[symbol]["balance"]:.2f}'
            }), 200

    # ─── Close out webhook() ─────────────────────────────────────────────────
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/balance', methods=['GET'])
def check_balance():
    """Endpoint to check current ticker balances"""
    try:
        ticker_balances = load_ticker_balances()
        trade_log = load_trade_log()

        # Calculate totals
        total_balance = sum(data['balance'] for data in ticker_balances.values())
        total_invested = sum(data['total_invested'] for data in ticker_balances.values())
        total_realized_pnl = sum(data['total_realized_pnl'] for data in ticker_balances.values())

        return jsonify({
            'ticker_balances': ticker_balances,
            'trade_log': trade_log,
            'summary': {
                'total_balance': f'${total_balance:.2f}',
                'total_invested': f'${total_invested:.2f}',
                'total_realized_pnl': f'${total_realized_pnl:.2f}',
                'initial_balance_per_ticker': f'${INITIAL_BALANCE_PER_TICKER}',
                'balance_usage_percent': f'{BALANCE_USAGE_PERCENT * 100}%'
            }
        }), 200
    except Exception as e:
        logger.error(f"Error checking balance: {e}")
        return jsonify({'error': 'Failed to check balance'}), 500


@app.route('/sync/<symbol>', methods=['POST'])
def sync_position_endpoint(symbol):
    """Endpoint to manually sync a specific ticker position"""
    try:
        ticker_balances = load_ticker_balances()
        actual_shares = sync_position_with_alpaca(symbol.upper(), ticker_balances)

        return jsonify({
            'symbol': symbol.upper(),
            'actual_shares': actual_shares,
            'sync_completed': True,
            'message': f'Position synced for {symbol.upper()}'
        }), 200
    except Exception as e:
        logger.error(f"Error syncing position for {symbol}: {e}")
        return jsonify({'error': f'Failed to sync position for {symbol}'}), 500


@app.route('/log/<symbol>', methods=['GET'])
def get_trade_log(symbol):
    """Endpoint to get trade log for a specific ticker"""
    try:
        trade_log = load_trade_log()
        symbol_log = trade_log.get(symbol.upper(), {})

        return jsonify({
            'symbol': symbol.upper(),
            'trade_log': symbol_log
        }), 200
    except Exception as e:
        logger.error(f"Error getting trade log for {symbol}: {e}")
        return jsonify({'error': f'Failed to get trade log for {symbol}'}), 500


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'Webhook server is running!'}), 200

@app.route('/cron-ping', methods=['GET'])
def cron_ping():
    return "pong", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)




import os
import asyncio
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile
from dotenv import load_dotenv

# ---------------------- НАСТРОЙКИ ----------------------

EXCHANGES = [
    "binance",
    "okx",
    "bybit",
    "coinbase",
    "kraken",
    "bitfinex",
    "kucoin",
    "huobi",
    "gateio",
    "bitstamp",
    "mexc",
    "bitget",
    "cryptocom",
    "bingx",
    "whitebit",
    "lbank",
    "upbit",
    "bithumb",
    "poloniex",
    "bitmart",
]

TIMEFRAME = "1h"
LIMIT = 300          # сколько свечей брать с каждой биржи
ROLLING_WINDOW = 24  # окно для расчета волатильности (24 часа)

OUTPUT_DIR = "output"

# ---------------------- УТИЛИТЫ ----------------------


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_ohlcv_from_exchanges(asset: str) -> pd.DataFrame:
    """
    Загружает OHLCV-данные по заданному активу (BTC или ETH)
    c набора бирж в EXCHANGES и возвращает общий DataFrame
    с MultiIndex-колонками: (биржа, ['close', 'volume']).
    """
    frames = []

    for ex_id in EXCHANGES:
        try:
            exchange_class = getattr(ccxt, ex_id)
        except AttributeError:
            # такой биржи нет в ccxt
            continue

        exchange = exchange_class({"enableRateLimit": True})
        try:
            markets = exchange.load_markets()
        except Exception as e:
            print(f"[{ex_id}] не удалось загрузить рынки: {e}")
            continue

        symbol = None
        for quote in ["USDT", "USDC", "USD"]:
            candidate = f"{asset}/{quote}"
            if candidate in markets:
                symbol = candidate
                break

        if not symbol:
            print(f"[{ex_id}] не найден подходящий рынок для {asset}")
            continue

        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
        except Exception as e:
            print(f"[{ex_id}] ошибка fetch_ohlcv: {e}")
            continue

        if not ohlcv:
            print(f"[{ex_id}] пустые данные OHLCV")
            continue

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)

        df = df[["close", "volume"]]
        df.columns = pd.MultiIndex.from_product([[ex_id], df.columns])
        frames.append(df)

    if not frames:
        raise RuntimeError("Не удалось получить данные ни с одной биржи.")

    combined = pd.concat(frames, axis=1).sort_index()
    return combined


def compute_vix_series(combined_df: pd.DataFrame) -> pd.Series:
    """
    Считает VIX-подобный индекс (реализованная волатильность)
    на основе объёмно-взвешенной цены (аналог VWAP по закрытию).

    Формула:
    1) Для каждого часа: VWAP_close = sum(p_i * v_i) / sum(v_i)
    2) Доходности: r_t = ln(P_t / P_{t-1})
    3) Реализованная волатильность на окне N:
       sigma_t = std(r_{t-N+1} ... r_t) * sqrt(24 * 365)
    4) Индекс VIX = sigma_t * 100 (в процентах годовых)
    """
    # получаем матрицы цен и объемов
    closes = combined_df.xs("close", level=1, axis=1)
    volumes = combined_df.xs("volume", level=1, axis=1)

    # VWAP (по закрытию)
    total_volume = volumes.sum(axis=1)
    vwap_price = (closes * volumes).sum(axis=1) / total_volume

    # логарифмические доходности
    log_returns = np.log(vwap_price / vwap_price.shift(1))

    # реализованная волатильность в годовых, % (аналог VIX)
    annual_factor = 24 * 365  # 24 часа * 365 дней
    realized_vol = log_returns.rolling(ROLLING_WINDOW).std() * np.sqrt(annual_factor)
    vix_series = realized_vol * 100.0

    return vix_series.dropna(), vwap_price


def save_to_excel(
    asset: str,
    combined_df: pd.DataFrame,
    vix_series: pd.Series,
    vwap_price: pd.Series,
) -> str:
    """
    Сохраняет данные в Excel: сырой массив по биржам, VWAP и индекс VIX.
    """
    ensure_output_dir()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"vix_{asset.lower()}_{timestamp}.xlsx"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # немного подготовим данные
    combined_export = combined_df.copy()
    combined_export.index.name = "datetime"

    vwap_df = pd.DataFrame(
        {
            "datetime": vwap_price.index,
            "vwap_price": vwap_price.values,
        }
    ).set_index("datetime")

    vix_df = pd.DataFrame(
        {
            "datetime": vix_series.index,
            "vix_index": vix_series.values,
        }
    ).set_index("datetime")

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        combined_export.to_excel(writer, sheet_name=f"{asset}_raw_exchanges")
        vwap_df.to_excel(writer, sheet_name=f"{asset}_VWAP")
        vix_df.to_excel(writer, sheet_name=f"{asset}_VIX")

    return filepath


def save_plot(asset: str, vix_series: pd.Series) -> str:
    """
    Строит график VIX-подобного индекса и сохраняет в PNG.
    """
    ensure_output_dir()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"vix_{asset.lower()}_{timestamp}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    plt.figure(figsize=(10, 5))
    plt.plot(vix_series.index, vix_series.values)
    plt.title(f"VIX-подобный индекс вынужденных продаж для {asset}")
    plt.xlabel("Дата и время (UTC)")
    plt.ylabel("Индекс VIX (годовая волатильность, %)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath


def calculate_vix_for_asset(asset: str):
    """
    Полный цикл:
    1) загрузить данные с бирж,
    2) посчитать индекс,
    3) сохранить Excel и график.
    """
    combined_df = fetch_ohlcv_from_exchanges(asset)
    vix_series, vwap_price = compute_vix_series(combined_df)
    excel_path = save_to_excel(asset, combined_df, vix_series, vwap_price)
    plot_path = save_plot(asset, vix_series)
    return excel_path, plot_path, vix_series.iloc[-1]


# ---------------------- TELEGRAM-БОТ ----------------------


load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    raise RuntimeError(
        "Не задан TELEGRAM_BOT_TOKEN. Создайте файл .env и укажите туда токен."
    )

logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: Message):
    text = (
        "Привет! Я бот индекса принудительных продаж (VIX) по Bitcoin и Ethereum.\n\n"
        "Команды:\n"
        "/vix_btc — посчитать и прислать индекс VIX для Bitcoin\n"
        "/vix_eth — посчитать и прислать индекс VIX для Ethereum\n"
        "/vix_all — сразу оба актива\n\n"
        "При каждом запросе я:\n"
        "• Собираю данные с топ-20 бирж (через ccxt)\n"
        "• Считаю VIX-подобный индекс (годовая реализованная волатильность)\n"
        "• Присылаю Excel-файл с данными и PNG-график."
    )
    await message.answer(text)


async def handle_vix_command(message: Message, asset: str):
    await message.answer(
        f"Запускаю анализ {asset} по топ-20 биржам. Это может занять немного времени..."
    )

    loop = asyncio.get_running_loop()
    try:
        excel_path, plot_path, last_vix = await loop.run_in_executor(
            None, calculate_vix_for_asset, asset
        )
    except Exception as e:
        logging.exception("Ошибка при расчете VIX")
        await message.answer(f"Ошибка при расчете индекса VIX для {asset}: {e}")
        return

    # Отправляем Excel
    excel_file = FSInputFile(excel_path)
    await message.answer_document(
        excel_file,
        caption=f"Данные для {asset}: индекс VIX и исходные данные по биржам.",
    )

    # Отправляем график
    plot_file = FSInputFile(plot_path)
    await message.answer_photo(
        plot_file,
        caption=(
            f"График VIX-подобного индекса для {asset}.\n\n"
            f"Текущее значение индекса (последняя точка): {last_vix:.2f} % годовой волатильности."
        ),
    )


@dp.message(Command("vix_btc"))
async def cmd_vix_btc(message: Message):
    await handle_vix_command(message, "BTC")


@dp.message(Command("vix_eth"))
async def cmd_vix_eth(message: Message):
    await handle_vix_command(message, "ETH")


@dp.message(Command("vix_all"))
async def cmd_vix_all(message: Message):
    await handle_vix_command(message, "BTC")
    await handle_vix_command(message, "ETH")


async def main():
    ensure_output_dir()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
